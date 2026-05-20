import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
import re
import concurrent.futures
from urllib.parse import urljoin
import json
import hashlib

# --- ページ設定とデザイン ---
st.set_page_config(page_title="車両画像チェックツール", layout="centered", page_icon="🚗")

st.markdown("""
<style>
    h3 { color: #333333; font-weight: 600; margin-top: 1.5rem; }
    hr { margin-top: 2rem; margin-bottom: 2rem; border-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🚗 車両画像チェックツール")
st.markdown("カーセンサーの掲載ページURLとローカルの画像を比較し、未掲載の画像のみを抽出します。")
st.markdown("---")

st.markdown("### ① カーセンサーの物件ページURL")
page_url = st.text_input("URL", label_visibility="collapsed", placeholder="例: https://www.carsensor.net/usedcar/detail/...")

st.markdown("### ② ローカルファイル")
st.caption("比較したい車両画像をアップロードしてください（複数選択・ZIPファイル対応）")
local_files = st.file_uploader("ファイルを選択", label_visibility="collapsed", type=['zip', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)

# 詳細設定
with st.expander("⚙️ 詳細設定（マッチング感度）"):
    sensitivity = st.select_slider(
        "判定感度",
        options=["厳しめ（見逃し減）", "標準", "緩め（誤検出減）"],
        value="標準",
        help="「厳しめ」は未掲載と判定されにくく、「緩め」は未掲載と判定されやすくなります"
    )

# 感度パラメータマップ
SENSITIVITY_MAP = {
    "厳しめ（見逃し減）": {"ratio": 0.82, "min_inliers": 8,  "min_inlier_ratio": 0.10, "phash_threshold": 20, "orb_match_threshold": 20},
    "標準":              {"ratio": 0.75, "min_inliers": 10, "min_inlier_ratio": 0.12, "phash_threshold": 15, "orb_match_threshold": 15},
    "緩め（誤検出減）":   {"ratio": 0.68, "min_inliers": 15, "min_inlier_ratio": 0.18, "phash_threshold": 10, "orb_match_threshold": 10},
}

# =============================================================================
# 画像取得・前処理
# =============================================================================

def resize_image(img, max_width=1200):
    """アスペクト比を維持してリサイズ（解像度を上げて特徴量精度向上）"""
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)
    return img


def normalize_image(img_gray):
    """CLAHEでコントラスト正規化（明るさ・圧縮差に頑健）"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def compute_phash(img_gray, hash_size=16):
    """知覚ハッシュ（pHash）を計算。色や圧縮差に頑健"""
    try:
        resized = cv2.resize(img_gray, (hash_size * 2, hash_size * 2), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:hash_size, :hash_size]
        median = np.median(dct_low)
        return (dct_low > median).flatten()
    except Exception:
        return None


def compute_dhash(img_gray, hash_size=16):
    """差分ハッシュ（dHash）。pHashと組み合わせて精度向上"""
    try:
        resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()
    except Exception:
        return None


def hamming_distance(h1, h2):
    if h1 is None or h2 is None:
        return 9999
    return int(np.count_nonzero(h1 != h2))


def combined_hash_distance(ph1, dh1, ph2, dh2):
    """pHash と dHash を組み合わせたハッシュ距離（重み付き）"""
    pd = hamming_distance(ph1, ph2)
    dd = hamming_distance(dh1, dh2)
    # pHashを重視（3:1）
    return (pd * 3 + dd) // 4


# =============================================================================
# カーセンサー専用：URL変換・画像抽出
# =============================================================================

def upgrade_carsensor_url(url):
    """
    カーセンサーのサムネイルURLを大画像URLに変換。
    ccsrpcma CDNのサイズパターンに対応。
    """
    candidates = [url]

    # パターン1: _200.jpg → _800.jpg, _400.jpg → _800.jpg 等の数値サイズ
    def replace_size(m):
        size = int(m.group(1))
        if size < 800:
            return m.group(0).replace(m.group(1), '800')
        return m.group(0)
    new_url = re.sub(r'_(\d{2,4})\.(jpe?g|png)', replace_size, url, flags=re.IGNORECASE)
    if new_url != url:
        candidates.insert(0, new_url)  # 大画像を優先

    # パターン2: /S/ → /L/, /M/ → /L/ 等のディレクトリ名
    for pat, rep in [
        (r'_S(\.(jpe?g|png))', r'_L\1'),
        (r'_M(\.(jpe?g|png))', r'_L\1'),
        (r'_s(\.(jpe?g|png))', r'_l\1'),
        (r'/S/', '/L/'),
        (r'/M/', '/L/'),
        (r'/thumb/', '/large/'),
        (r'thumb', 'large'),
        (r'small', 'large'),
    ]:
        u2 = re.sub(pat, rep, url, flags=re.IGNORECASE)
        if u2 != url and u2 not in candidates:
            candidates.append(u2)

    return candidates


FETCH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Referer': 'https://www.carsensor.net/',
}


def fetch_image(url):
    """画像をダウンロードしグレースケールで返す（大画像優先）"""
    for candidate_url in upgrade_carsensor_url(url):
        try:
            res = requests.get(candidate_url, headers=FETCH_HEADERS, timeout=8)
            if res.status_code == 200 and len(res.content) > 5000:
                nparr = np.frombuffer(res.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    if h >= 150 and w >= 150:
                        aspect = w / h
                        if 0.3 < aspect < 4.0:
                            return resize_image(img)
        except Exception:
            continue
    return None


def extract_image_urls_from_html(html_text, base_url):
    """
    HTMLから車両画像URLを多段階で抽出。
    カーセンサーのJSON埋め込み・遅延読み込みに対応。
    """
    urls = set()

    # 1. JSON-LD 構造化データ
    for ld_block in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html_text, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(ld_block)
            text = json.dumps(data)
            for u in re.findall(r'https?://[^"\'<>\s]+\.(?:jpg|jpeg|png)', text, re.IGNORECASE):
                urls.add(u)
        except Exception:
            pass

    # 2. OGP メタタグ
    for u in re.findall(r'<meta[^>]+(?:og:image|twitter:image)[^>]+content=["\']([^"\']+)["\']', html_text, re.IGNORECASE):
        urls.add(u)
    for u in re.findall(r'content=["\']([^"\']+\.(?:jpg|jpeg|png)[^"\']*)["\'][^>]+property=["\']og:image["\']', html_text, re.IGNORECASE):
        urls.add(u)

    # 3. カーセンサー専用：JS変数・JSONオブジェクト内の画像URL
    # 例: "photo_url":"https://..."  "img_src":"..."
    for key in ['photo_url', 'img_src', 'image_url', 'src', 'url', 'photo', 'image']:
        for u in re.findall(rf'["\']?{key}["\']?\s*:\s*["\']([^"\']+\.(?:jpg|jpeg|png)[^"\']*)["\']', html_text, re.IGNORECASE):
            urls.add(u)

    # 4. 通常の img src
    for u in re.findall(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png)[^"\']*)["\']', html_text, re.IGNORECASE):
        urls.add(u)

    # 5. 遅延読み込み属性
    for attr in ['data-src', 'data-original', 'data-lazy', 'data-lazy-src', 'data-echo', 'data-img-src']:
        for u in re.findall(rf'{attr}=["\']([^"\']+\.(?:jpg|jpeg|png)[^"\']*)["\']', html_text, re.IGNORECASE):
            urls.add(u)

    # 6. srcset
    for srcset in re.findall(r'srcset=["\']([^"\']+)["\']', html_text, re.IGNORECASE):
        for part in srcset.split(','):
            u = part.strip().split(' ')[0]
            if re.search(r'\.(jpg|jpeg|png)', u, re.IGNORECASE):
                urls.add(u)

    # 7. JSON文字列内のURL（広範囲）
    for u in re.findall(r'["\']((?:https?:)?//[^"\'<>\s]+\.(?:jpg|jpeg|png)(?:\?[^"\'<>\s]*)?)["\']', html_text, re.IGNORECASE):
        urls.add(u)

    # 正規化・フィルタリング
    filtered = set()
    for u in urls:
        if u.startswith('//'):
            u = 'https:' + u
        elif u.startswith('/'):
            u = urljoin(base_url, u)
        elif not u.startswith('http'):
            continue

        u_lower = u.lower()

        # 車両画像ドメイン/パスのみ許可
        if not any(k in u_lower for k in ['carsensor', 'ccsrpcma', 'picture', 'photo', 'image', 'img']):
            continue

        # 不要なUI画像を除外
        if any(ng in u_lower for ng in [
            'logo', 'icon', 'banner', 'btn_', 'sprite', 'spacer', 'blank',
            'common/', 'arrow', 'star', 'badge', 'tag', 'label', 'mark',
            'bg_', '_bg', 'loading', 'noimage', 'no_image', 'dummy'
        ]):
            continue

        filtered.add(u)

    return filtered


def get_images_from_url(url):
    """ページから車両画像を取得（並列ダウンロード・重複除去付き）"""
    try:
        page_headers = {
            'User-Agent': FETCH_HEADERS['User-Agent'],
            'Accept-Language': 'ja,en;q=0.9',
        }
        response = requests.get(url, headers=page_headers, timeout=15)
        response.raise_for_status()
        html_text = response.text

    except Exception as e:
        st.error(f"ページの読み込みに失敗しました。({e})")
        return None

    image_urls = extract_image_urls_from_html(html_text, url)

    if not image_urls:
        st.warning("ページから画像URLが見つかりませんでした。")
        return []

    # 並列ダウンロード（スレッド数を増やして高速化）
    web_gray_images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_image, u): u for u in image_urls}
        for future in concurrent.futures.as_completed(futures):
            img = future.result()
            if img is not None:
                web_gray_images.append(img)

    # 重複除去（pHash + dHash で二重チェック）
    unique_images = []
    unique_hashes = []  # (phash, dhash)
    for img in web_gray_images:
        normalized = normalize_image(img)
        ph = compute_phash(normalized)
        dh = compute_dhash(normalized)
        is_dup = False
        for uph, udh in unique_hashes:
            if combined_hash_distance(ph, dh, uph, udh) <= 5:
                is_dup = True
                break
        if not is_dup:
            unique_images.append(img)  # 正規化前を保存（特徴量計算は別途正規化）
            unique_hashes.append((ph, dh))

    return unique_images


# =============================================================================
# 特徴量マッチング
# =============================================================================

def compute_sift_features(img_normalized, sift):
    """SIFT特徴量を計算（CLAHE正規化済み画像を使用）"""
    kp, des = sift.detectAndCompute(img_normalized, None)
    if des is None or len(des) < 8:
        return None
    return {'kp': kp, 'des': des}


def compute_orb_features(img_normalized, orb):
    """ORB特徴量を計算（SIFTの補完用・高速）"""
    kp, des = orb.detectAndCompute(img_normalized, None)
    if des is None or len(des) < 8:
        return None
    return {'kp': kp, 'des': des}


def match_with_sift(local_feat, web_feat, flann, params):
    """SIFT + RANSAC マッチング"""
    if local_feat is None or web_feat is None:
        return False, 0

    des_l, des_w = local_feat['des'], web_feat['des']
    if len(des_l) < 2 or len(des_w) < 2:
        return False, 0

    try:
        matches = flann.knnMatch(des_l, des_w, k=2)
    except cv2.error:
        return False, 0

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < params['ratio'] * n.distance:
                good.append(m)

    if len(good) < params['min_inliers']:
        return False, len(good)

    src_pts = np.float32([local_feat['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([web_feat['kp'][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return False, 0

    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(good) if good else 0

    if inliers >= params['min_inliers'] and inlier_ratio >= params['min_inlier_ratio']:
        if M is not None:
            det = np.linalg.det(M[:2, :2])
            if 0.05 < abs(det) < 20:  # スケール許容範囲を拡大
                return True, inliers

    return False, inliers


def match_with_orb(local_feat, web_feat, params):
    """ORB + BF マッチング（SIFTが失敗した場合の補完）"""
    if local_feat is None or web_feat is None:
        return False, 0

    des_l, des_w = local_feat['des'], web_feat['des']
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        matches = bf.knnMatch(des_l, des_w, k=2)
    except cv2.error:
        return False, 0

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    threshold = params['orb_match_threshold']
    if len(good) >= threshold:
        return True, len(good)

    return False, len(good)


# =============================================================================
# 並列特徴量事前計算
# =============================================================================

def precompute_web_features(web_images, sift, orb):
    """Web画像の特徴量・ハッシュを並列計算"""
    def compute_one(img):
        normalized = normalize_image(img)
        ph = compute_phash(normalized)
        dh = compute_dhash(normalized)
        sf = compute_sift_features(normalized, sift)
        of = compute_orb_features(normalized, orb)
        return ph, dh, sf, of

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(compute_one, img) for img in web_images]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # 順序を保持（as_completedは順不同なのでインデックス付きで再実行）
    ordered = [None] * len(web_images)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {executor.submit(compute_one, img): i for i, img in enumerate(web_images)}
        for f in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[f]
            ordered[idx] = f.result()

    return ordered  # List of (phash, dhash, sift_feat, orb_feat)


# =============================================================================
# メイン処理
# =============================================================================

def process_images(web_images, local_file_objs, params):
    # --- 検出器の初期化 ---
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=15, sigma=1.6)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    index_params = dict(algorithm=1, trees=8)      # KD-Tree
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # --- Web画像の特徴量を並列事前計算 ---
    with st.spinner("Web画像の特徴量を計算中..."):
        web_feats = precompute_web_features(web_images, sift, orb)
    # web_feats[i] = (phash, dhash, sift_feat, orb_feat)

    # --- ローカル画像を展開 ---
    image_data_list = []
    for uploaded_file in local_file_objs:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    for zinfo in z.infolist():
                        name = zinfo.filename
                        if zinfo.is_dir():
                            continue
                        if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        if '__MACOSX' in name or name.startswith('.'):
                            continue
                        image_data_list.append((name, z.read(name)))
            except zipfile.BadZipFile:
                st.warning(f"{uploaded_file.name} は壊れたZIPファイルです")
        else:
            image_data_list.append((uploaded_file.name, file_bytes))

    if not image_data_list:
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(image_data_list)
    missing_images = []

    for i, (file_name, file_bytes) in enumerate(image_data_list):
        nparr = np.frombuffer(file_bytes, np.uint8)
        local_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if local_img is None:
            progress_bar.progress((i + 1) / total)
            continue

        local_resized = resize_image(local_img)
        local_norm = normalize_image(local_resized)
        local_ph = compute_phash(local_norm)
        local_dh = compute_dhash(local_norm)
        local_sift = compute_sift_features(local_norm, sift)
        local_orb = compute_orb_features(local_norm, orb)

        is_found = False

        # ====================================================
        # Step 1: 即時マッチ（ハッシュが非常に近い → 確定）
        # ====================================================
        for ph, dh, _, _ in web_feats:
            dist = combined_hash_distance(local_ph, local_dh, ph, dh)
            if dist <= 5:
                is_found = True
                break

        if is_found:
            status_text.caption(f"処理中: {i + 1}/{total} - {file_name} ✅ (ハッシュ一致)")
            progress_bar.progress((i + 1) / total)
            continue

        # ====================================================
        # Step 2: pHash距離でソートして候補を絞り込み
        # ====================================================
        candidates = []
        for idx, (ph, dh, sf, of) in enumerate(web_feats):
            dist = combined_hash_distance(local_ph, local_dh, ph, dh)
            candidates.append((dist, idx))
        candidates.sort(key=lambda x: x[0])

        # pHash閾値内の候補を優先、なければ上位N件にフォールバック
        priority = [idx for d, idx in candidates if d <= params['phash_threshold']]
        if not priority:
            # 閾値外でもSIFTは上位10件まで試す（フォールバック）
            priority = [idx for _, idx in candidates[:10]]

        # ====================================================
        # Step 3: SIFT + RANSAC（精密マッチング）
        # ====================================================
        for idx in priority:
            _, _, sf, of = web_feats[idx]
            matched, score = match_with_sift(local_sift, sf, flann, params)
            if matched:
                is_found = True
                break

            # SIFT失敗 → ORBで補完
            if not matched and of is not None and local_orb is not None:
                orb_matched, orb_score = match_with_orb(local_orb, of, params)
                if orb_matched:
                    is_found = True
                    break

        if not is_found:
            missing_images.append((file_name, file_bytes))

        status_text.caption(f"処理中: {i + 1}/{total} - {file_name}")
        progress_bar.progress((i + 1) / total)

    status_text.empty()
    return missing_images


# =============================================================================
# UI: 比較実行
# =============================================================================

st.markdown("---")
st.markdown("### ③ 画像の比較を開始する")

if page_url and local_files:
    if st.button("✨ 比較を実行する", use_container_width=True, type="primary"):
        params = SENSITIVITY_MAP[sensitivity]

        with st.spinner("サイトから画像を抽出しています..."):
            web_images = get_images_from_url(page_url)

        if web_images is not None:
            if len(web_images) == 0:
                st.warning("サイトから画像を取得できませんでした。URLを確認してください。")
            else:
                st.caption(f"※サイトから {len(web_images)} 枚のユニーク画像を取得しました")

                with st.spinner("画像を比較しています..."):
                    missing_list = process_images(web_images, local_files, params)

                # --- ④ 掲載のない画像のダウンロード ---
                st.markdown("---")
                st.markdown("### ④ 掲載のない画像のダウンロード")

                if missing_list is None:
                    st.error("比較できるローカル画像が見つかりませんでした。")
                elif missing_list:
                    st.info(f"未掲載の画像が **{len(missing_list)}** 枚見つかりました。")

                    with st.expander("🖼 未掲載画像のプレビュー"):
                        cols = st.columns(3)
                        for idx, (fname, fbytes) in enumerate(missing_list[:12]):
                            with cols[idx % 3]:
                                st.image(fbytes, caption=fname.split('/')[-1], use_container_width=True)
                        if len(missing_list) > 12:
                            st.caption(f"...他 {len(missing_list) - 12} 枚")

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for file_name, file_bytes in missing_list:
                            safe_name = file_name.split("/")[-1]
                            zip_file.writestr(safe_name, file_bytes)

                    st.download_button(
                        label="📥 ZIPファイルでダウンロード",
                        data=zip_buffer.getvalue(),
                        file_name="missing_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.success("🎉 すべてのローカル画像が掲載ページに存在します！")
else:
    st.info("💡 ①と②のデータをセットすると、比較ボタンが押せるようになります。")