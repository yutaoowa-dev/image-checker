import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="車両画像チェックツール", layout="centered")

st.title("🚗 車両画像チェックツール")
st.write("カーセンサーの掲載ページURLとローカルの画像を比較し、未掲載の画像をZIPでダウンロードします。")

st.success("### ①カーセンサーの物件ページURL")
page_url = st.text_input("URLを貼り付けてください", placeholder="例: https://www.carsensor.net/usedcar/detail/...")

st.success("### ②ローカルファイル")
local_files = st.file_uploader("ローカルの車両画像（ZIPファイル、または複数画像）", type=['zip', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)

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
                    web_gray_images.append(img_gray)
            except Exception:
                continue
        return web_gray_images
    except Exception as e:
        st.error(f"ページの読み込みに失敗しました。({e})")
        return None

def process_images(web_images, local_file_objs):
    orb = cv2.ORB_create()
    
    web_des_list = []
    for img in web_images:
        _, des = orb.detectAndCompute(img, None)
        if des is not None:
            web_des_list.append(des)
            
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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

        _, des_local = orb.detectAndCompute(local_gray, None)
        
        is_found = False
        if des_local is not None:
            for des_web in web_des_list:
                matches = bf.match(des_local, des_web)
                good_matches = [m for m in matches if m.distance < 50]
                if len(good_matches) >= 30: 
                    is_found = True
                    break
        
        if not is_found:
            missing_images.append((file_name, file_bytes))
            
        progress_bar.progress((i + 1) / total)

    return missing_images

st.markdown("---")
st.error("### ③掲載のない画像のダウンロード")

if page_url and local_files:
    if st.button("画像の比較を開始する", use_container_width=True):
        with st.spinner("掲載ページから画像を取得し、比較しています..."):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                missing_list = process_images(web_images, local_files)
                
                if missing_list is None:
                    st.error("比較できるローカル画像が見つかりませんでした。")
                elif missing_list:
                    st.info(f"掲載されていない画像が {len(missing_list)} 枚見つかりました！")
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for file_name, file_bytes in missing_list:
                            safe_name = file_name.split("/")[-1] 
                            zip_file.writestr(safe_name, file_bytes)
                    
                    st.download_button(
                        label="📥 未掲載画像をZIPでダウンロード",
                        data=zip_buffer.getvalue(),
                        file_name="missing_images.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                else:
                    st.success("すべてのローカル画像が掲載ページに存在します！")
else:
    st.warning("URLの入力と画像のアップロードが完了すると、ボタンが表示されます。")