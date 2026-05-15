import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
import re
import concurrent.futures

# --- ページ設定 ---
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

# --- 画像処理関数 ---

def resize_image(img, max_width=800): # 精度を出すために少し大きめにリサイズ
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def fetch_image(url):
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            nparr = np.frombuffer(res.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                if h > 150 and w > 150: # アイコン除外の基準を少し上げました
                    return resize_image(img)
    except:
        pass
    return None

def get_images_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_text = response.text
        
        pattern = r'(?:https?:)?//[a-zA-Z0-9\-\./_]+(?:\.jpg|\.jpeg|\.png)'
        all_urls = re.findall(pattern, html_text, re.IGNORECASE)
        
        web_images_urls = set()
        for src in all_urls:
            # 物件画像によく使われるパスを含むものに限定（ノイズ除去）
            if 'carsensor' in src or 'picture' in src or 'contents' in src:
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
    sift = cv2.SIFT_create(nfeatures=2000) # 特徴点抽出数を増やして精度UP
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    web_features = []
    for img in web_images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None and len(des) > 20:
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
        if des_local is not None and len(des_local) > 20:
            for web_feat in web_features:
                if len(web_feat['des']) < 2:
                    continue
                    
                matches = flann.knnMatch(des_local, web_feat['des'], k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # 判定基準を0.75から0.65に厳格化（曖昧な一致を排除）
                        if m.distance < 0.65 * n.distance:
                            good_matches.append(m)
                
                # RANSACによる幾何学チェック
                if len(good_matches) >= 20: # 最低一致数を20に引き上げ
                    src_pts = np.float32([kp_local[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([web_feat['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        inliers = np.sum(mask)
                        # 一致とみなす基準を20箇所に大幅引き上げ
                        if inliers >= 20:
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
    if st.button("✨ 比較を実行する（厳格モード）", use_container_width=True, type="primary"):
        with st.spinner("最高精度の厳格判定を行っています。少し時間がかかります..."):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                st.caption(f"※サイトから {len(web_images)} 枚の画像データを取得しました")
                
                missing_list = process_images(web_images, local_files)
                
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
                        label="📥 未掲載分のみZIPでダウンロード",
                        data=zip_buffer.getvalue(),
                        file_name="missing_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.success("🎉 すべてのローカル画像が掲載ページに存在します！")
else:
    st.info("💡 ①と②のデータをセットすると、比較ボタンが押せるようになります。")