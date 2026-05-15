import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import requests
from bs4 import BeautifulSoup

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
    """比較しやすいサイズに揃える"""
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def calc_dhash(img_gray):
    """画像の大まかな構造をハッシュ化する（画質やサイズの違いに非常に強い）"""
    resized = cv2.resize(img_gray, (9, 8))
    # 隣り合うピクセルの明るさを比較して真偽値の配列を作る
    return resized[:, 1:] > resized[:, :-1]

def get_images_from_url(url):
    """URLから画像を漏れなく取得する"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        web_images = []
        for img in soup.find_all('img'):
            # srcだけでなく、遅延読み込み用の属性(data-srcなど)も探す
            src = img.get('data-src') or img.get('data-original') or img.get('src')
            if src and ('carsensor' in src or 'picture' in src or 'car' in src):
                if src.startswith('//'):
                    src = 'https:' + src
                web_images.append(src)
                
        web_gray_images = []
        # 重複するURLを排除
        for img_url in set(web_images):
            try:
                res = requests.get(img_url, timeout=5)
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
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Web画像の特徴(AKAZE)と構造(dHash)を両方計算して保存
    web_features = []
    for img in web_images:
        _, des = akaze.detectAndCompute(img, None)
        img_hash = calc_dhash(img)
        web_features.append({'des': des, 'hash': img_hash})
            
    missing_images = []
    image_data_list = []
    
    # アップロードされたファイルを展開
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
        hash_local = calc_dhash(local_gray)
        
        is_found = False
        for web_feat in web_features:
            # 【判定1】dHashによる構図の一致チェック（違いが12箇所/64箇所以下なら同じとみなす）
            hash_diff = np.sum(hash_local != web_feat['hash'])
            if hash_diff <= 12:
                is_found = True
                break
                
            # 【判定2】AKAZEによる特徴点チェック（判定基準を前回の15から10へ緩和）
            if des_local is not None and web_feat['des'] is not None and len(des_local) > 2 and len(web_feat['des']) > 2:
                try:
                    matches = bf.knnMatch(des_local, web_feat['des'], k=2)
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            # ここも0.75から0.8に緩和し、よりマッチしやすくした
                            if m.distance < 0.8 * n.distance:
                                good_matches.append(m)
                                
                    if len(good_matches) >= 10: 
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
        with st.spinner("サイトからすべての画像を収集し、AIで比較しています..."):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                # 取得した枚数をこっそり表示（デバッグ用・何枚取れたか確認できます）
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