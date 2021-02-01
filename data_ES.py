import os
import csv
import glob
import numpy as np
import collections
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize



# Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

def atributos_2da_prediccion(df_new):
    
    df_new['SlopeClose_1'] = (df_new['Close_1'] - df_new['Close_1'].shift(1))/1
    df_new['SlopeClose_5'] = (df_new['Close_1'] - df_new['Close_1'].shift(5))/5
    df_new['SlopeClose_35'] = (df_new['Close_1'] - df_new['Close_1'].shift(35))/35
    
    df_new['SlopeOpen_1'] = (df_new['Open_1'] - df_new['Open_1'].shift(1))/1
    df_new['SlopeOpen_5'] = (df_new['Open_1'] - df_new['Open_1'].shift(5))/5
    df_new['SlopeOpen_35'] = (df_new['Open_1'] - df_new['Open_1'].shift(35))/35
    
    df_new['SlopeLow_1'] = (df_new['Low_1'] - df_new['Low_1'].shift(1))/1
    df_new['SlopeLow_5'] = (df_new['Low_1'] - df_new['Low_1'].shift(5))/5
    df_new['SlopeLow_35'] = (df_new['Low_1'] - df_new['Low_1'].shift(35))/35
    
    df_new['SlopeHigh_1'] = (df_new['High_1'] - df_new['High_1'].shift(1))/1
    df_new['SlopeHigh_5'] = (df_new['High_1'] - df_new['High_1'].shift(5))/5
    df_new['SlopeHigh_35'] = (df_new['High_1'] - df_new['High_1'].shift(35))/35
    
    df_new['SlopeVolume_1'] = (df_new['Volume_1'] - df_new['Volume_1'].shift(1))/1
    df_new['SlopeVolume_5'] = (df_new['Volume_1'] - df_new['Volume_1'].shift(5))/5
    df_new['SlopeVolume_35'] = (df_new['Volume_1'] - df_new['Volume_1'].shift(35))/35    
    
    df_new.fillna(value=0,inplace=True)
    
    return df_new

def generacion_atributos(df_new):
    
    # 6 original features
    df_new['Open_1'] = df_new['Open'].shift(1)
    df_new['Close_1'] = df_new['Close'].shift(1)
    df_new['High_1'] = df_new['High'].shift(1)
    df_new['Low_1'] = df_new['Low'].shift(1)
    df_new['Volume_1'] = df_new['Volume'].shift(1)
    df_new = df_new.drop(["Date_Minute","High","Low","Close","Volume"],axis=1)
    return df_new

def generacion_atributos_2(df_new):
    # 31 original features
    # average price
    #----------------Porcentaje tipo de barra-------------------#
    
    #--------------Promedios y ratios Close-----------------------
    
    #df_new['avg_price_5'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).mean()
    #df_new['avg_price_15'] = pd.Series(df_new[parametro]).shift(1).rolling(window=15).mean()
    #df_new['avg_price_60'] = pd.Series(df_new[parametro]).shift(1).rolling(window=60).mean()
    """Sin shift xq ya se corrieronn los valores y se trabaja con Close_1"""
    
    # df_new['avg_price_5'] = pd.Series(df_new[parametro]).rolling(window=5).mean()
    # df_new['avg_price_15'] = pd.Series(df_new[parametro]).rolling(window=15).mean()
    # df_new['avg_price_60'] = pd.Series(df_new[parametro]).rolling(window=60).mean()
    
    # df_new['ratio_avg_price_5_15'] = df_new['avg_price_5'] / df_new['avg_price_15']
    # df_new['ratio_avg_price_5_60'] = df_new['avg_price_5'] / df_new['avg_price_60']
    
    #--------------AT-----------------------------------------------
    #df_new['Elliot'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).mean() - pd.Series(df_new[parametro]).shift(1).rolling(window=35).mean()
    df_new['Elliot'] = pd.Series(df_new['Close_1']).rolling(window=5).mean() - pd.Series(df_new['Close_1']).rolling(window=35).mean()

    
    df_new['dif_max_vs_min'] = df_new['High_1'].shift(1) - df_new['Low_1'].shift(1)
    df_new['dif_open_close'] = df_new['Open_1'].shift(1) - df_new['Close_1'].shift(1)
    #df_new['ATR'] = pd.Series(df_new['Close_1'].shift(1) - df_new['Open_1'].shift(1)).rolling(window=5).mean()
    df_new = df_new.drop(["Open_1"],axis=1)     
    
    #----------------------------- average volume------------------------
    #df_new['avg_volume_5'] = pd.Series(df_new['Volume']).shift(1).rolling(window=5).mean()
    #df_new['avg_volume_15'] = pd.Series(df_new['Volume']).shift(1).rolling(window=15).mean()
    #df_new['avg_volume_35'] = pd.Series(df_new['Volume']).shift(1).rolling(window=35).mean()
    #df_new['avg_volume_60'] = pd.Series(df_new['Volume']).shift(1).rolling(window=60).mean()
    
    # df_new['avg_volume_5'] = pd.Series(df_new['Volume_1']).rolling(window=5).mean()
    # df_new['avg_volume_15'] = pd.Series(df_new['Volume_1']).rolling(window=15).mean()
    # df_new['avg_volume_35'] = pd.Series(df_new['Volume_1']).rolling(window=35).mean()
    # df_new['avg_volume_60'] = pd.Series(df_new['Volume_1']).rolling(window=60).mean()
    
    # df_new['dif_avg_volume_5_35'] = df_new['avg_volume_5'] - df_new['avg_volume_35']

    #---------------------------- standard deviation of prices
    #df_new['std_price_5'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).std()
    #df_new['std_price_15'] = pd.Series(df_new[parametro]).shift(1).rolling(window=15).std()
    #df_new['std_price_60'] = pd.Series(df_new[parametro]).shift(1).rolling(window=60).std()
    #df_new['std_ratio_avg_volume_5_30'] = pd.Series(df_new['dif_avg_volume_5_35']).shift(1).rolling(window=15).std()
    
    # df_new['std_price_5'] = pd.Series(df_new[parametro]).rolling(window=5).std()
    # df_new['std_price_15'] = pd.Series(df_new[parametro]).rolling(window=15).std()
    # df_new['std_price_60'] = pd.Series(df_new[parametro]).rolling(window=60).std()
    # df_new['std_ratio_avg_volume_5_30'] = pd.Series(df_new['dif_avg_volume_5_35']).rolling(window=15).std()
    
    # df_new['ratio_std_price_5_15'] = df_new['std_price_5'] / df_new['std_price_15']
    # df_new['ratio_std_price_5_60'] = df_new['std_price_5'] / df_new['std_price_60']
    # df_new['ratio_std_price_15_60'] = df_new['std_price_15'] / df_new['std_price_60']
    
    #----------------------------- standard deviation of volumes
    #df_new['std_volume_5'] = pd.Series(df_new['Volume']).shift(1).rolling(window=5).std()
    #df_new['std_volume_15'] = pd.Series(df_new['Volume']).shift(1).rolling(window=15).std()
    #df_new['std_volume_60'] = pd.Series(df_new['Volume']).shift(1).rolling(window=60).std()
    
    # df_new['std_volume_5'] = pd.Series(df_new['Volume_1']).rolling(window=5).std()
    # df_new['std_volume_15'] = pd.Series(df_new['Volume_1']).rolling(window=15).std()
    # df_new['std_volume_60'] = pd.Series(df_new['Volume_1']).rolling(window=60).std()
    # df_new['ratio_std_volume_5_15'] = df_new['std_volume_5'] / df_new['std_volume_15']
    # df_new['ratio_std_volume_5_60'] = df_new['std_volume_5'] / df_new['std_volume_60']
    # df_new['ratio_std_volume_15_60'] = df_new['std_volume_15'] / df_new['std_volume_60']
    
    # # # return
    # df_new['return_1'] = (((df_new[parametro] - df_new[parametro].shift(1)) / df_new[parametro].shift(1)).shift(1))*100
    # df_new['return_5'] = (((df_new[parametro] - df_new[parametro].shift(5)) / df_new[parametro].shift(5)).shift(1))*100
    # df_new['return_15'] = (((df_new[parametro] - df_new[parametro].shift(15)) / df_new[parametro].shift(15)).shift(1))*100
    # df_new['return_60'] = (((df_new[parametro] - df_new[parametro].shift(60)) / df_new[parametro].shift(60)).shift(1))*100
    # df_new['moving_avg_return_5'] = pd.Series(df_new['return_1']).rolling(window=5).mean()
    # df_new['moving_avg_return_15'] = pd.Series(df_new['return_1']).rolling(window=15).mean()
    # df_new['moving_avg_return_60'] = pd.Series(df_new['return_1']).rolling(window=60).mean()
    
    # the target
    return df_new

def generacion_atributos_2_features(df_new):
    # 31 original features
    # average price
    #----------------Porcentaje tipo de barra-------------------#
    
    #--------------Promedios y ratios Close-----------------------
    
    #df_new['avg_price_5'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).mean()
    #df_new['avg_price_15'] = pd.Series(df_new[parametro]).shift(1).rolling(window=15).mean()
    #df_new['avg_price_60'] = pd.Series(df_new[parametro]).shift(1).rolling(window=60).mean()
    """Sin shift xq ya se corrieronn los valores y se trabaja con Close_1"""
    
    # df_new['avg_price_5'] = pd.Series(df_new[parametro]).rolling(window=5).mean()
    # df_new['avg_price_15'] = pd.Series(df_new[parametro]).rolling(window=15).mean()
    # df_new['avg_price_60'] = pd.Series(df_new[parametro]).rolling(window=60).mean()
    
    # df_new['ratio_avg_price_5_15'] = df_new['avg_price_5'] / df_new['avg_price_15']
    # df_new['ratio_avg_price_5_60'] = df_new['avg_price_5'] / df_new['avg_price_60']
    
    #--------------AT-----------------------------------------------
    #df_new['Elliot'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).mean() - pd.Series(df_new[parametro]).shift(1).rolling(window=35).mean()
    df_new['Elliot'] = pd.Series(df_new['Close_1']).rolling(window=5).mean() - pd.Series(df_new['Close_1']).rolling(window=35).mean()

    
    #df_new['dif_max_vs_min'] = df_new['High_1'].shift(1) - df_new['Low_1'].shift(1)
    #df_new['dif_open_close'] = df_new['Open_1'].shift(1) - df_new['Close_1'].shift(1)
          
    
    #----------------------------- average volume------------------------
    #df_new['avg_volume_5'] = pd.Series(df_new['Volume']).shift(1).rolling(window=5).mean()
    #df_new['avg_volume_15'] = pd.Series(df_new['Volume']).shift(1).rolling(window=15).mean()
    #df_new['avg_volume_35'] = pd.Series(df_new['Volume']).shift(1).rolling(window=35).mean()
    #df_new['avg_volume_60'] = pd.Series(df_new['Volume']).shift(1).rolling(window=60).mean()
    
    # df_new['avg_volume_5'] = pd.Series(df_new['Volume_1']).rolling(window=5).mean()
    # df_new['avg_volume_15'] = pd.Series(df_new['Volume_1']).rolling(window=15).mean()
    # df_new['avg_volume_35'] = pd.Series(df_new['Volume_1']).rolling(window=35).mean()
    # df_new['avg_volume_60'] = pd.Series(df_new['Volume_1']).rolling(window=60).mean()
    
    # df_new['dif_avg_volume_5_35'] = df_new['avg_volume_5'] - df_new['avg_volume_35']

    #---------------------------- standard deviation of prices
    #df_new['std_price_5'] = pd.Series(df_new[parametro]).shift(1).rolling(window=5).std()
    #df_new['std_price_15'] = pd.Series(df_new[parametro]).shift(1).rolling(window=15).std()
    #df_new['std_price_60'] = pd.Series(df_new[parametro]).shift(1).rolling(window=60).std()
    #df_new['std_ratio_avg_volume_5_30'] = pd.Series(df_new['dif_avg_volume_5_35']).shift(1).rolling(window=15).std()
    
    # df_new['std_price_5'] = pd.Series(df_new[parametro]).rolling(window=5).std()
    # df_new['std_price_15'] = pd.Series(df_new[parametro]).rolling(window=15).std()
    # df_new['std_price_60'] = pd.Series(df_new[parametro]).rolling(window=60).std()
    # df_new['std_ratio_avg_volume_5_30'] = pd.Series(df_new['dif_avg_volume_5_35']).rolling(window=15).std()
    
    # df_new['ratio_std_price_5_15'] = df_new['std_price_5'] / df_new['std_price_15']
    # df_new['ratio_std_price_5_60'] = df_new['std_price_5'] / df_new['std_price_60']
    # df_new['ratio_std_price_15_60'] = df_new['std_price_15'] / df_new['std_price_60']
    
    #----------------------------- standard deviation of volumes
    #df_new['std_volume_5'] = pd.Series(df_new['Volume']).shift(1).rolling(window=5).std()
    #df_new['std_volume_15'] = pd.Series(df_new['Volume']).shift(1).rolling(window=15).std()
    #df_new['std_volume_60'] = pd.Series(df_new['Volume']).shift(1).rolling(window=60).std()
    
    # df_new['std_volume_5'] = pd.Series(df_new['Volume_1']).rolling(window=5).std()
    # df_new['std_volume_15'] = pd.Series(df_new['Volume_1']).rolling(window=15).std()
    # df_new['std_volume_60'] = pd.Series(df_new['Volume_1']).rolling(window=60).std()
    # df_new['ratio_std_volume_5_15'] = df_new['std_volume_5'] / df_new['std_volume_15']
    # df_new['ratio_std_volume_5_60'] = df_new['std_volume_5'] / df_new['std_volume_60']
    # df_new['ratio_std_volume_15_60'] = df_new['std_volume_15'] / df_new['std_volume_60']
    
    # # # return
    # df_new['return_1'] = (((df_new[parametro] - df_new[parametro].shift(1)) / df_new[parametro].shift(1)).shift(1))*100
    # df_new['return_5'] = (((df_new[parametro] - df_new[parametro].shift(5)) / df_new[parametro].shift(5)).shift(1))*100
    # df_new['return_15'] = (((df_new[parametro] - df_new[parametro].shift(15)) / df_new[parametro].shift(15)).shift(1))*100
    # df_new['return_60'] = (((df_new[parametro] - df_new[parametro].shift(60)) / df_new[parametro].shift(60)).shift(1))*100
    # df_new['moving_avg_return_5'] = pd.Series(df_new['return_1']).rolling(window=5).mean()
    # df_new['moving_avg_return_15'] = pd.Series(df_new['return_1']).rolling(window=15).mean()
    # df_new['moving_avg_return_60'] = pd.Series(df_new['return_1']).rolling(window=60).mean()
    
    # the target
    return df_new


def read_csv(file_name, sep=',', filter_data=True, fix_open_price=False):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_name, ';')
        indices = [h.index(s) for s in ('<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>')]
        o, h, l, c, v = [], [], [], [], []
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None
        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and all(map(lambda v: abs(v-vals[0]) < 1e-8, vals[:-1])):
                count_filter += 1
                continue

            po, ph, pl, pc, pv = vals

            # fix open price for current bar to match close price for the previous bar
            if fix_open_price and prev_vals is not None:
                ppo, pph, ppl, ppc, ppv = prev_vals
                if abs(po - ppc) > 1e-8:
                    count_fixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)
            count_out += 1
            o.append(po)
            c.append(pc)
            h.append(ph)
            l.append(pl)
            v.append(pv)
            prev_vals = vals
    print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
        count_filter + count_out, count_filter, count_fixed))
    return Prices(open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32),
                  volume=np.array(v, dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result

def read_csv_ES(ruta, sep=',', filter_data=True, fix_open_price=False):
    
    datos = pd.read_csv(ruta,sep=sep,
                        names= ["Date_Minute","Open","High","Low","Close","Volume"],
                        engine='c',parse_dates=["Date_Minute"],
                        infer_datetime_format=True)
 
        
    datos_prueba = generacion_atributos(datos)
    datos_prueba = generacion_atributos_2(datos_prueba)
    prices = datos_prueba
    
    
    return prices
    
    
    # return Prices(open=np.array(o, dtype=np.float32),
    #               high=np.array(h, dtype=np.float32),
    #               low=np.array(l, dtype=np.float32),
    #               close=np.array(c, dtype=np.float32),
    #               volume=np.array(v, dtype=np.float32))



def prices_to_relative_ES(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    print(type(prices))
    # assert isinstance(prices, pd.DataFrame)
    # rh = (prices.High - prices.open) / prices.open
    # rl = (prices.Low - prices.open) / prices.open
    # rc = (prices.Close - prices.open) / prices.open
    return prices

def load_relative_ES(csv_file):
    return prices_to_relative_ES(read_csv_ES(csv_file))

