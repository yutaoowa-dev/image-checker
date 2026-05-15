import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
from bs4 import BeautifulSoup
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
RESIZE_WIDTH = 400  # 比較処理を軽くするために少し小さくリサイズ

def resize_image(img, max_width=RESIZE_WIDTH):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def calc_dhash(img_gray, hash_size=16):
    """画像の大まかな構造をハッシュ化（16x16の256箇所で厳密に比較し、誤検知を防ぐ）"""
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def fetch_image(url):
    """並列処理で画像を高速ダウンロードするための関数"""
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            nparr = np.frombuffer(res.content, np.uint8)
            img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img_gray is not None:
                return resize_image(img_gray)
    except:
        pass
    return None

def get_images_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        web_images_urls = set()
        for img in soup.find_all('img'):
            src = img.get('data-src') or img.get('data-original') or img.get('src')
            # 車両画像が置いてあるドメインで絞り込み
            if src and ('carsensor.net' in src or 'picture' in src):
                if src.startswith('//'):
                    src = 'https:' + src
                web_images_urls.add(src)
                
        web_gray_images = []
        
        # ★ ここが高速化の鍵！数十枚の画像を同時にダウンロードします
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
    # 重いAKAZEをやめ、高速なORBに変更
    orb = cv2.ORB_create(nfeatures=500) 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    web_features = []
    for img in web_images:
        _, des = orb.detectAndCompute(img, None)
        img_hash = calc_dhash(img)
        web_features.append({'des': des, 'hash': img_hash})
            
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
        _, des_local = orb.detectAndCompute(local_gray, None)
        hash_local = calc_dhash(local_gray)
        
        is_found = False
        for web_feat in web_features:
            # 【判定1】高解像度dHashによる厳密チェック
            hash_diff = np.sum(hash_local != web_feat['hash'])
            # 256bit中、違いが12以下なら「同じ画像」とみなす（約95%一致）
            if hash_diff <= 12:
                is_found = True
                break
                
            # 【判定2】ORB特徴点による厳密チェック
            if des_local is not None and web_feat['des'] is not None:
                try:
                    matches = bf.match(des_local, web_feat['des'])
                    # 距離が近い（似ている）特徴点だけを厳選
                    good_matches = [m for m in matches if m.distance < 45]
                    # 車の写真は似やすいので、一致する点が多い場合（30個以上）のみ同じとみなす
                    if len(good_matches) >= 30: 
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
        with st.spinner("サイトから80枚以上の画像を高速取得し、AIで比較しています..."):
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