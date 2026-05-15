import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
from bs4 import BeautifulSoup

# --- ページ設定 ---
st.set_page_config(page_title="車両画像チェックツール", layout="centered", page_icon="🚗")

# --- カスタムCSS（シンプルなデザインにするための微調整） ---
st.markdown("""
<style>
    /* 見出しの色と太さを調整 */
    h3 {
        color: #333333;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    /* 区切り線を薄く控えめに */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# --- タイトルエリア ---
st.title("🚗 車両画像チェックツール")
st.markdown("カーセンサーの掲載ページURLとローカルの画像を比較し、未掲載の画像のみを抽出します。")
st.markdown("---")

# --- ① URL入力 ---
st.markdown("### ① カーセンサーの物件ページURL")
page_url = st.text_input("URL", label_visibility="collapsed", placeholder="例: https://www.carsensor.net/usedcar/detail/...")

# --- ② ファイルアップロード ---
st.markdown("### ② ローカルファイル")
st.caption("比較したい車両画像をアップロードしてください（複数選択・ZIPファイル対応）")
local_files = st.file_uploader("ファイルを選択", label_visibility="collapsed", type=['zip', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)

# --- 画像処理の関数群（前回の高精度版のまま） ---
def resize_image(img, max_width=600):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def get_images_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        web_images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and ('ccsrpcma.carsensor.net' in src or 'picture' in src):
                if src.startswith('//'):
                    src = 'https:' + src
                web_images.append(src)
                
        web_gray_images = []
        for img_url in set(web_images):
            try:
                res = requests.get(img_url)
                nparr = np.frombuffer(res.content, np.uint8)
                img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    img_gray = resize_image(img_gray)
                    web_gray_images.append(img_gray)
            except Exception:
                continue
        return web_gray_images
    except Exception as e:
        st.error(f"ページの読み込みに失敗しました。({e})")
        return None

def process_images(web_images, local_file_objs):
    akaze = cv2.AKAZE_create()
    
    web_des_list = []
    for img in web_images:
        _, des = akaze.detectAndCompute(img, None)
        if des is not None:
            web_des_list.append(des)
            
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
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
        local_gray = cv2.imdecode(nparr_local, cv2.IMREAD_GRAYSCALE)

        if local_gray is None:
            continue
            
        local_gray = resize_image(local_gray)
        _, des_local = akaze.detectAndCompute(local_gray, None)
        
        is_found = False
        if des_local is not None and len(des_local) > 2:
            for des_web in web_des_list:
                if des_web is None or len(des_web) < 2:
                    continue
                    
                try:
                    matches = bf.knnMatch(des_local, des_web, k=2)
                    
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                                
                    if len(good_matches) >= 15: 
                        is_found = True
                        break
                except Exception:
                    continue
        
        if not is_found:
            missing_images.append((file_name, file_bytes))
            
        progress_bar.progress((i + 1) / total)

    return missing_images

# --- ③ 画像の比較を開始する ---
st.markdown("---")
st.markdown("### ③ 画像の比較を開始する")

if page_url and local_files:
    if st.button("✨ 比較を実行する", use_container_width=True, type="primary"):
        with st.spinner("掲載ページから画像を取得し、比較しています...（少し時間がかかります）"):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                missing_list = process_images(web_images, local_files)
                
                # --- ④ 掲載のない画像のダウンロード（処理が終わったら表示） ---
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