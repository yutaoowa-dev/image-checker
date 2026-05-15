# File: app.py
import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
import re
import concurrent.futures
from urllib.parse import urljoin

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
    "厳しめ（見逃し減）": {"ratio": 0.80, "min_inliers": 8,  "min_inlier_ratio": 0.10, "phash_threshold": 18},
    "標準":              {"ratio": 0.75, "min_inliers": 12, "min_inlier_ratio": 0.15, "phash_threshold": 14},
    "緩め（誤検出減）":   {"ratio": 0.70, "min_inliers": 18, "min_inlier_ratio": 0.20, "phash_threshold": 10},
}

# --- 画像処理の関数群 ---

def resize_image(img, max_width=1000):
    """アスペクト比を維持してリサイズ"""
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)
    return img


def compute_phash(img_gray, hash_size=16):
    """知覚ハッシュ（pHash）を計算。色や圧縮差に頑健"""
    try:
        # 32x32にリサイズしてDCT
        resized = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        # 左上の低周波成分を取得
        dct_low = dct[:hash_size, :hash_size]
        median = np.median(dct_low)
        return (dct_low > median).flatten()
    except Exception:
        return None


def hamming_distance(h1, h2):
    if h1 is None or h2 is None:
        return 9999
    return int(np.count_nonzero(h1 != h2))


def upgrade_carsensor_url(url):
    """カーセンサーのサムネイルURLを大画像URLに変換する試み"""
    # 例: _S.jpg → _L.jpg, /S/ → /L/, thumb → large 等
    candidates = [url]
    # 末尾サイズ識別子の変換
    patterns = [
        (r'_S(\.(jpe?g|png))', r'_L\1'),
        (r'_M(\.(jpe?g|png))', r'_L\1'),
        (r'_s(\.(jpe?g|png))', r'_l\1'),
        (r'/S/', '/L/'),
        (r'/M/', '/L/'),
        (r'thumb', 'large'),
        (r'small', 'large'),
    ]
    for pat, rep in patterns:
        new_url = re.sub(pat, rep, url)
        if new_url != url and new_url not in candidates:
            candidates.append(new_url)
    return candidates


def fetch_image(url):
    """画像をダウンロードしグレースケール＋カラーで返す"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        # 大画像版を優先して試す
        for candidate_url in upgrade_carsensor_url(url):
            try:
                res = requests.get(candidate_url, headers=headers, timeout=8)
                if res.status_code == 200 and len(res.content) > 3000:
                    nparr = np.frombuffer(res.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h, w = img.shape[:2]
                        # アイコン・バナー類は除外
                        if h >= 150 and w >= 150:
                            aspect = w / h
                            # 極端に細長い画像（バナー）も除外
                            if 0.3 < aspect < 4.0:
                                return resize_image(img)
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_images_from_url(url):
    """ページから車両画像URLを抽出（遅延読み込み対応）"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'ja,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_text = response.text

        web_images_urls = set()

        # 1. 通常の画像URL抽出
        pattern_normal = r'(?:https?:)?//[a-zA-Z0-9\-\./_%]+\.(?:jpg|jpeg|png)(?:\?[^"\'\s<>]*)?'
        for src in re.findall(pattern_normal, html_text, re.IGNORECASE):
            web_images_urls.add(src)

        # 2. data-src, data-original, data-lazy 等の遅延読み込み属性
        pattern_lazy = r'data-(?:src|original|lazy|lazy-src|echo)=["\']([^"\']+\.(?:jpg|jpeg|png)[^"\']*)["\']'
        for m in re.findall(pattern_lazy, html_text, re.IGNORECASE):
            web_images_urls.add(m)

        # 3. srcset 属性内の画像
        pattern_srcset = r'srcset=["\']([^"\']+)["\']'
        for srcset in re.findall(pattern_srcset, html_text, re.IGNORECASE):
            for part in srcset.split(','):
                u = part.strip().split(' ')[0]
                if re.search(r'\.(jpg|jpeg|png)', u, re.IGNORECASE):
                    web_images_urls.add(u)

        # 4. JSON内の画像URL（カーセンサーはJSON埋込が多い）
        pattern_json = r'["\']((?:https?:)?//[^"\']+\.(?:jpg|jpeg|png))["\']'
        for m in re.findall(pattern_json, html_text, re.IGNORECASE):
            web_images_urls.add(m)

        # フィルタリング & 正規化
        filtered_urls = set()
        for src in web_images_urls:
            if src.startswith('//'):
                src = 'https:' + src
            elif src.startswith('/'):
                src = urljoin(url, src)
            elif not src.startswith('http'):
                continue

            src_lower = src.lower()
            # 車両画像と思われるドメイン/パスのみ
            if any(k in src_lower for k in ['carsensor', 'ccsrpcma', 'picture', 'photo', 'image']):
                # 明らかに不要なものを除外
                if not any(ng in src_lower for ng in ['logo', 'icon', 'banner', 'btn_', 'sprite', 'spacer', 'blank', 'common/']):
                    filtered_urls.add(src)

        web_gray_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results = executor.map(fetch_image, filtered_urls)
            for img in results:
                if img is not None:
                    web_gray_images.append(img)

        # 重複除去（pHashで近似重複を排除）
        unique_images = []
        unique_hashes = []
        for img in web_gray_images:
            ph = compute_phash(img)
            is_dup = False
            for uh in unique_hashes:
                if hamming_distance(ph, uh) <= 4:
                    is_dup = True
                    break
            if not is_dup:
                unique_images.append(img)
                unique_hashes.append(ph)

        return unique_images
    except Exception as e:
        st.error(f"ページの読み込みに失敗しました。({e})")
        return None


def compute_sift_features(img, sift):
    """SIFT特徴量を計算"""
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) < 10:
        return None
    return {'kp': kp, 'des': des, 'shape': img.shape}


def match_with_sift(local_feat, web_feat, flann, params):
    """SIFT+RANSACで2画像をマッチング判定。一致なら(True, inliers)を返す"""
    if local_feat is None or web_feat is None:
        return False, 0

    des_local = local_feat['des']
    des_web = web_feat['des']

    if len(des_local) < 2 or len(des_web) < 2:
        return False, 0

    try:
        matches = flann.knnMatch(des_local, des_web, k=2)
    except cv2.error:
        return False, 0

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < params['ratio'] * n.distance:
                good_matches.append(m)

    if len(good_matches) < params['min_inliers']:
        return False, len(good_matches)

    src_pts = np.float32([local_feat['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([web_feat['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return False, 0

    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(good_matches) if good_matches else 0

    # 絶対数 AND 比率の両方を満たす
    if inliers >= params['min_inliers'] and inlier_ratio >= params['min_inlier_ratio']:
        # ホモグラフィの妥当性チェック（極端な変形を除外）
        if M is not None:
            det = np.linalg.det(M[:2, :2])
            if 0.1 < abs(det) < 10:  # 極端なスケール変化を除外
                return True, inliers

    return False, inliers


def process_images(web_images, local_file_objs, params):
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03, edgeThreshold=15)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Web画像の特徴量とpHashを事前計算
    web_features = []
    web_phashes = []
    for img in web_images:
        feat = compute_sift_features(img, sift)
        ph = compute_phash(img)
        web_features.append(feat)
        web_phashes.append(ph)

    # ローカル画像を展開
    image_data_list = []
    for uploaded_file in local_file_objs:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    for zinfo in z.infolist():
                        if not zinfo.is_dir() and zinfo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # macOSのメタファイル除外
                            if '__MACOSX' in zinfo.filename or zinfo.filename.startswith('.'):
                                continue
                            image_data_list.append((zinfo.filename, z.read(zinfo.filename)))
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
        nparr_local = np.frombuffer(file_bytes, np.uint8)
        local_img = cv2.imdecode(nparr_local, cv2.IMREAD_GRAYSCALE)

        if local_img is None:
            progress_bar.progress((i + 1) / total)
            continue

        local_resized = resize_image(local_img)
        local_phash = compute_phash(local_resized)
        local_feat = compute_sift_features(local_resized, sift)

        is_found = False
        best_score = 0

        # Step1: pHashで近似判定（高速）
        phash_candidates = []
        for idx, web_ph in enumerate(web_phashes):
            dist = hamming_distance(local_phash, web_ph)
            if dist <= params['phash_threshold']:
                phash_candidates.append((idx, dist))

        # pHashが非常に近い場合は即マッチ確定
        for idx, dist in phash_candidates:
            if dist <= 6:
                is_found = True
                best_score = 1000 - dist
                break

        # Step2: SIFTで厳密判定（pHash候補を優先、なければ全件）
        if not is_found and local_feat is not None:
            # 候補順を構築：pHash近い順 + その他
            candidate_order = [idx for idx, _ in sorted(phash_candidates, key=lambda x: x[1])]
            remaining = [k for k in range(len(web_features)) if k not in candidate_order]
            search_order = candidate_order + remaining

            for idx in search_order:
                matched, score = match_with_sift(local_feat, web_features[idx], flann, params)
                if score > best_score:
                    best_score = score
                if matched:
                    is_found = True
                    break

        if not is_found:
            missing_images.append((file_name, file_bytes))

        status_text.caption(f"処理中: {i + 1}/{total} - {file_name}")
        progress_bar.progress((i + 1) / total)

    status_text.empty()
    return missing_images


# --- ③ 画像の比較を開始する ---
st.markdown("---")
st.markdown("### ③ 画像の比較を開始する")

if page_url and local_files:
    if st.button("✨ 比較を実行する", use_container_width=True, type="primary"):
        params = SENSITIVITY_MAP[sensitivity]

        with st.spinner("サイトから画像を抽出しています..."):
            web_images = get_images_from_url(page_url)

        if web_images:
            st.caption(f"※サイトから {len(web_images)} 枚のユニーク画像を取得しました")

            with st.spinner("pHash + SIFT + RANSAC で厳密に比較しています..."):
                missing_list = process_images(web_images, local_files, params)

            # --- ④ 掲載のない画像のダウンロード ---
            st.markdown("---")
            st.markdown("### ④ 掲載のない画像のダウンロード")

            if missing_list is None:
                st.error("比較できるローカル画像が見つかりませんでした。")
            elif missing_list:
                st.info(f"未掲載の画像が **{len(missing_list)}** 枚見つかりました。")

                # プレビュー表示
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