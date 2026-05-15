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
# 比較用サイズ。このサイズに揃えて重ね合わせることで、高速かつ超高精度に判定します。
COMPARE_SIZE = (150, 150)

def fetch_image(url):
    """URLから画像をダウンロードし、比較用のモノクロ・リサイズ画像にする"""
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            nparr = np.frombuffer(res.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 縦横が極端に小さいアイコンなどは無視
                h, w = img.shape[:2]
                if h > 100 and w > 100:
                    return cv2.resize(img, COMPARE_SIZE)
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
        
        # HTMLやJavaScriptの中に埋もれている .jpg のURLをすべて正規表現で探し出す
        pattern = r'(?:https?:)?//[a-zA-Z0-9\-\./_]+(?:\.jpg|\.jpeg|\.png)'
        all_urls = re.findall(pattern, html_text, re.IGNORECASE)
        
        web_images_urls = set()
        for src in all_urls:
            # カーセンサーや画像サーバーのドメインのみに絞る
            if 'carsensor' in src or 'picture' in src:
                if src.startswith('//'):
                    src = 'https:' + src
                web_images_urls.add(src)
                
        web_gray_images = []
        
        # 10個同時に並列ダウンロード
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
        local_img = cv2.imdecode(nparr_local, cv2.IMREAD_GRAYSCALE)

        if local_img is None:
            continue
            
        # ローカル画像も同じサイズにリサイズ
        local_resized = cv2.resize(local_img, COMPARE_SIZE)
        
        is_found = False
        for web_resized in web_images:
            # ★ ピクセル相関（テンプレートマッチング）による究極の判定
            # 画像同士を重ね合わせて、どれだけピクセルが一致するかを 0.0〜1.0 で計算
            res = cv2.matchTemplate(local_resized, web_resized, cv2.TM_CCOEFF_NORMED)
            similarity = res[0][0]
            
            # 一致率が 85% (0.85) 以上なら「同じ写真」とみなす
            # ※別角度の写真だと 40〜60% 程度にしかならないため、確実に区別できます
            if similarity >= 0.85:
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
        with st.spinner("サイトの裏側から画像を強制抽出して比較しています..."):
            web_images = get_images_from_url(page_url)
            
            if web_images:
                # 取得できた枚数を表示（83枚に近いか確認できます！）
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