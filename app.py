from datetime import datetime, timedelta
import os

#import altair as alt
#import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.title("日本プライム株-可視化アプリ")

# 取得期間の指定
days = 30

@st.cache_resource #キャッシュに貯めておくことで早く読み取ることができる

#銘柄リストから買い増し推奨銘柄を抽出するための関数
def get_data(days, stock_namelist_df):
    df = pd.DataFrame()
    
    for index, company in stock_namelist_df.iterrows():
        tkr = yf.Ticker(company['銘柄コード'])
        stock_data = tkr.history(period=f'{days}d')
        stock_data.index = stock_data.index.strftime('%d %B %Y')

        ### RSI

        # RSIを計算する
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        stock_data["RSI"] = rsi
        stock_data["RSI"] = stock_data["RSI"].fillna(50)

        # RSI係数
        stock_data["RSI_Long_index"] = 0
        for i in range(13, len(stock_data)):
            RSI_min = stock_data['RSI'][i-13:i+1].min()
            if RSI_min >= 42:
                RSI_index = 0.1
            elif RSI_min >= 30:
                RSI_index = 30/((RSI_min + 4 * stock_data['RSI'][i])/5)
            else:
                RSI_index = 1
            stock_data.iloc[i, stock_data.columns.get_loc("RSI_Long_index")] = RSI_index

        stock_data["RSI_Short_index"] = 0
        for i in range(13, len(stock_data)):
            RSI_max = stock_data['RSI'][i-13:i+1].max()
            if RSI_max <= 55:
                RSI_index = -0.1
            elif RSI_max <= 70:
                RSI_index = -30/((RSI_max + 4 * stock_data['RSI'][i])/5)
            else:
                RSI_index = -1
            stock_data.iloc[i, stock_data.columns.get_loc("RSI_Short_index")] = RSI_index

        ### MACD

        # MACDを計算するための期間を指定する
        macd_short = 12
        macd_long = 26
        macd_signal = 9

        # MACDを計算する
        ema_short = stock_data['Close'].ewm(span=macd_short, adjust=False).mean()
        ema_long = stock_data['Close'].ewm(span=macd_long, adjust=False).mean()
        stock_data['MACD'] = ema_short - ema_long
        stock_data['MACD_SIGNAL'] = stock_data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        stock_data['MACD_HISTGRAM'] = stock_data['MACD']-stock_data['MACD_SIGNAL'] 

        # MACDのクロスを検出する
        stock_data["MACD_cross"] = np.where(stock_data['MACD'] > stock_data['MACD_SIGNAL'], True, False)
        stock_data['MACD_GC_change'] = (stock_data["MACD_HISTGRAM"]  > 0) & (stock_data["MACD_cross"] != stock_data["MACD_cross"].shift())
        stock_data['MACD_DC_change'] = (stock_data["MACD_HISTGRAM"]  < 0) & (stock_data["MACD_cross"] != stock_data["MACD_cross"].shift())
        stock_data.loc[stock_data.index[0], 'MACD_GC_change'] = False
        stock_data.loc[stock_data.index[0], 'MACD_DC_change'] = False

        # MACDのゴールデンクロスからの日にちを追加
        stock_data["MACD_GC_days"] = 0
        count = 0
        for i, row in stock_data.iterrows():
            if row["MACD_GC_change"]:
                count = 1
            elif row["MACD_cross"]:
                count += 1
            elif (row["MACD_cross"] == False) & (row["MACD_GC_change"] == False):
                count = 0
            stock_data.loc[i, "MACD_GC_days"] = count

        # MACDのゴールデンクロス係数
        stock_data["MACD_GC_index"] = 0
        for i, row in stock_data.iterrows():
            if row["MACD"] < 0:
                if row["MACD_GC_days"] <= 0:
                    MACD_GC_index = 0.1
                elif row["MACD_GC_days"] <= 16:
                    MACD_GC_index =  -0.05 * row["MACD_GC_days"] + 1
                else:
                    MACD_GC_index = 0.2
            else:
                MACD_GC_index = 0.1
            stock_data.loc[i, "MACD_GC_index"] = MACD_GC_index

        # MACDのデッドクロスからの日にちを追加
        stock_data["MACD_DC_days"] = 0
        count = 0
        for i, row in stock_data.iterrows():
            if row["MACD_DC_change"]:
                count = 1
            elif not row["MACD_cross"]:
                count += 1
            elif (row["MACD_cross"] == True) & (row["MACD_DC_change"] == False):
                count = 0
            stock_data.loc[i, "MACD_DC_days"] = -count

        # MACDのデットクロス係数
        stock_data["MACD_DC_index"] = 0
        for i, row in stock_data.iterrows():
            if row["MACD"] > 0:
                if row["MACD_DC_days"] >= 0:
                    MACD_DC_index = -0.1
                elif row["MACD_DC_days"] >= -16:
                    MACD_DC_index =  -0.05 * row["MACD_DC_days"] - 1
                else:
                    MACD_DC_index = -0.2
            else:
                MACD_DC_index = -0.1
            stock_data.loc[i, "MACD_DC_index"] = MACD_DC_index

        ###　買い時指数
        #stock_data["buy_index"] = stock_data["MA_GC_index"] * stock_data["RSI_index"] * stock_data["MACD_GC_index"] 
        stock_data["buy_index"] = stock_data["RSI_Long_index"] * stock_data["MACD_GC_index"] 
        stock_data["sell_index"] = -stock_data["RSI_Short_index"] * stock_data["MACD_DC_index"] 

        buy_index_stock_data = stock_data[["Close", "Volume", "RSI", "MACD_GC_days", "MACD_DC_days", "buy_index", "sell_index"]]

        ### buy_and_sell_signal
        
        # keyの初期値を設定
        key = 1 # Long = 1、Short = 0
        buy_index_stock_data["buy_and_sell_signal"] = 0

        for i, row in buy_index_stock_data.iterrows():
            #Long key -> 1
            if key == 1:
                if row["buy_index"] > 0.90:
                    buy_and_sell_signal = 1
                    sell_price = row["Close"]
                    key = 1 - key

                else:
                    buy_and_sell_signal = 0
            #Short key -> 0
            elif key == 0:
                # 利確ライン
                if row["sell_index"] < -0.90:
                    buy_and_sell_signal = -1
                    key = 1 - key

                #　利確ライン2(購入から10%以上の上昇)
                elif row["Close"] / sell_price > 1.1:
                    buy_and_sell_signal = -1
                    profit_sell_price = row["Close"]
                    key = 1 - key

                # 損切りライン(購入から5%の減少)
                elif row["Close"] / sell_price < 0.95:
                    buy_and_sell_signal = -1
                    profit_sell_price = row["Close"]
                    key = 1 - key

                else:
                    buy_and_sell_signal = 0
            
            buy_index_stock_data.loc[i, "buy_and_sell_signal"] = buy_and_sell_signal

        ### 最新日のデータを抜き出して整形
        latest_stock_data = buy_index_stock_data.iloc[-1]
        latest_stock_data = pd.DataFrame(latest_stock_data).T
        latest_stock_data = latest_stock_data.rename_axis('Date').reset_index()
        
        latest_stock_data.index=[company['銘柄名']]

        company = pd.DataFrame(company)
        company = company.T
        company = company.set_index('銘柄名')

        latest_stock_data = pd.concat([company, latest_stock_data], axis=1)
        latest_stock_data.index.name = '銘柄名'

        df = pd.concat([df, latest_stock_data])
    
    return df

#監視銘柄の読み出し
# ファイル選択ダイアログを開く
input_path_name = './銘柄コード一覧_2023-03.csv'
upload_file = input_path_name
# upload_file = st.file_uploader('ファイルをアップロードしてください。', type='csv')

if upload_file:
    df = pd.read_csv(upload_file)

    df = df.dropna(axis=1)

    # columns=コードが数字4桁のもののみ抽出
    df = df[df['コード'].astype(str).str.match('^\d{4}$')]
    # columns='市場・商品区分'が「プライム（内国株式）」のみを抽出
    df = df[df['市場・商品区分'].str.contains('プライム（内国株式）')]

    df = df.rename(columns={'33業種区分':'セクター名'})
    df = df.rename(columns={'コード':'銘柄コード'})
    df = df[['銘柄名', '銘柄コード', 'セクター名']]
    df['銘柄コード'] = df['銘柄コード'].astype(str) + '.T'
    stock_namelist_df = df.reset_index(drop=True)

    # リストの買い増し指数をget_data関数にて取得
    stock_list_df = get_data(days, stock_namelist_df)

    # データ整形
    stock_list_df = stock_list_df.round({'Close':1, 'RSI':2, 'buy_index':2, 'sell_index':2})
    stock_list_df['Volume'] =stock_list_df['Volume'].astype(int)
    stock_list_df["MACD_GC_days"] =stock_list_df["MACD_GC_days"].astype(int)
    stock_list_df["MACD_DC_days"] =stock_list_df["MACD_DC_days"].astype(int)
    stock_list_df["buy_and_sell_signal"] =stock_list_df["buy_and_sell_signal"].astype(int)

    st.write("### 株価 (JPY)", stock_list_df)

    # csvファイルとして保存
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")  # YYYY-MM-DD形式の文字列
    filename_with_extension = os.path.basename(input_path_name)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    result_dir = "./result"
    result_path = os.makedirs(result_dir, exist_ok=True)
    output_file_name = filename_without_extension +"_"+ date_string + ".csv"
    output_dir = os.path.join(result_path, output_file_name)

    stock_list_df.to_csv(output_dir)