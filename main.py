import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
import re
import concurrent.futures

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

# --- 画像処理の関数群 ---

def resize_image(img, max_width=600):
    """アスペクト比（縦横比）を崩さずにリサイズする"""
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def fetch_image(url):
    """並列処理で画像をダウンロード"""
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            nparr = np.frombuffer(res.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                if h > 100 and w > 100: # 小さすぎるアイコンは除外
                    return resize_image(img)
    except:
        pass
    return None

def get_images_from_url(url):
    """ページのソースコードからすべての画像URLを強制抽出する"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_text = response.text
        
        pattern = r'(?:https?:)?//[a-zA-Z0-9\-\./_]+(?:\.jpg|\.jpeg|\.png)'
        all_urls = re.findall(pattern, html_text, re.IGNORECASE)
        
        web_images_urls = set()
        for src in all_urls:
            if 'carsensor' in src or 'picture' in src:
                if src.startswith('//'):
                    src = 'https:' + src
                web_images_urls.add(src)
                
        web_gray_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_image, web_images_urls)
            for img in results:
                if img is not None:
                    web_gray_images.append(img)
                    
        return web_gray_images
    except Exception as e:
        st.error(f"ページの読み込みに失敗しました。({e})")
        return None

def process_images(web_images, local_file_objs):
    sift = cv2.SIFT_create()
    
    # 高速・高精度な特徴点マッチャーの設定
    index_params = dict(algorithm=1, trees=5) # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 事前にWeb画像の特徴点を計算して保存（処理の高速化）
    web_features = []
    for img in web_images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None and len(des) > 10:
            web_features.append({'kp': kp, 'des': des})
            
    missing_images = []
    image_data_list = []
    
    for uploaded_file in local_file_objs:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for zinfo in z.infolist():
                    if not zinfo.is_dir() and zinfo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_data_list.append((zinfo.filename, z.read(zinfo.filename)))
        else:
            image_data_list.append((uploaded_file.name, file_bytes))

    if not image_data_list:
        return None

    progress_bar = st.progress(0)
    total = len(image_data_list)

    for i, (file_name, file_bytes) in enumerate(image_data_list):
        nparr_local = np.frombuffer(file_bytes, np.uint8)
        local_img = cv2.imdecode(nparr_local, cv2.IMREAD_GRAYSCALE)

        if local_img is None:
            continue
            
        local_resized = resize_image(local_img)
        kp_local, des_local = sift.detectAndCompute(local_resized, None)
        
        is_found = False
        if des_local is not None and len(des_local) > 10:
            for web_feat in web_features:
                if len(web_feat['des']) < 2:
                    continue
                    
                matches = flann.knnMatch(des_local, web_feat['des'], k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # 0.75の比率で確実に似ている点だけを絞り込む
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                # ★ 空間的な幾何学チェック (RANSACアルゴリズム)
                # 単に似ている点が多いだけでなく、「点の配置」が矛盾していないか計算する
                if len(good_matches) >= 10:
                    src_pts = np.float32([kp_local[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([web_feat['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # 射影変換行列を計算し、矛盾する点（外れ値）を除外する
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        # 空間的にも正しいと判断された一致点の数
                        inliers = np.sum(mask)
                        # 10箇所以上、幾何学的に正しく一致していれば同一画像とみなす
                        if inliers >= 10:
                            is_found = True
                            break
        
        if not is_found:
            missing_images.append((file_name, file_bytes))
            
        progress_bar.progress((i + 1) / total)

    return missing_images

# --- ③ 画像の比較を開始する ---
st.markdown("---")
st.markdown("### ③ 画像の比較を開始する")

if page_url and local_files:
    if st.button("✨ 比較を実行する", use_container_width=True, type="primary"):
        with st.spinner("サイトから画像を抽出し、幾何学チェック（RANSAC）で厳密に比較しています..."):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                st.caption(f"※サイトから {len(web_images)} 枚の画像データを取得しました")
                
                missing_list = process_images(web_images, local_files)
                
                # --- ④ 掲載のない画像のダウンロード ---
                st.markdown("---")
                st.markdown("### ④ 掲載のない画像のダウンロード")
                
                if missing_list is None:
                    st.error("比較できるローカル画像が見つかりませんでした。")
                elif missing_list:
                    st.info(f"未掲載の画像が **{len(missing_list)}** 枚見つかりました。")
                    
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