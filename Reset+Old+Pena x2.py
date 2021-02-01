import gym
import gym.spaces
import os
import time
import copy
import numpy as np
import enum
import pandas as pd

from gym.utils import seeding
from gym.envs.registration import EnvSpec
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_ES import generacion_atributos,load_relative_ES
from data_ES import generacion_atributos_2
from joblib import dump, load
from typing import List, Optional, Tuple, Any

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy,LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2,A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

DEFAULT_BARS_COUNT = 1
DEFAULT_COMMISSION_PERC = 2.5
INITIAL_BALANCE = 3000
EPSILON = 1.2

GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
BATCH_SIZE = 64
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

"""
En esta version se penaliza la duracion excesiva del trade
Tambien calcula la recompensa como =100/max(5,self.steps)

"""

class Actions(enum.Enum):
    Buy = 0
    Sell = 1
    Close = 2
    Skip = 3

def read_csv_ES(ruta, sep=',',minutos=0,filter_data=True, fix_open_price=False,resample=False):
    
    datos = pd.read_csv(ruta,sep=sep,
                        names= ["Date_Minute","Open","High","Low","Close","Volume"],
                        engine='c',parse_dates=["Date_Minute"],
                        infer_datetime_format=True)
    if resample:
        datos = datos.set_index(['Date_Minute'])
        datos.index = datos.index.tz_localize('UTC').tz_convert('America/Argentina/Cordoba')
        periodo = str(minutos)+'min'
        datos = datos.resample(periodo,closed='right',label='right').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
        datos.dropna(how='any',inplace=True)
        datos.fillna(value=0,inplace=True)
        datos = datos.reset_index()
        
    datos_prueb = generacion_atributos(datos)
    datos_prueba = generacion_atributos_2(datos_prueb)
    prices = datos_prueba
    return prices

def prices_to_relative_ES(prices): 
    return prices

def load_relative_ES(csv_file,minutos,resample):
    #return prices_to_relative_ES(read_csv_ES(csv_file,minutos=5,resample=True))
    return prices_to_relative_ES(read_csv_ES(csv_file,minutos=minutos,resample=resample))

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,account_balance = INITIAL_BALANCE,
                 limit = 0.5,take_scaler = False,name_scalar = None,test=False,
                 reset_on_close=True,random_ofs_on_reset=True, reward_on_close=False):
        
        self.bars_count = bars_count
        self.commission = commission
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.account_balance_unrealized = account_balance
        self.media = (prices.Close_1 - prices.Close_1.shift(1)).rolling(window=10).mean()
        self.limit = limit
        self.prices_witout_changes = prices
        self.name = name_scalar
        self.test = test
        
        #Creo los escaladores para todos los precios y para la columna Close_1
        if not take_scaler:
            self._standarScaler = StandardScaler()
            self._standarClose = StandardScaler() 
            
            # self._standarScaler = MinMaxScaler()
            # self._standarClose = MinMaxScaler() 
            
            self._prices = prices - prices.shift(1)
            self._prices = self._standarScaler.fit_transform(self._prices)
            self._prices = np.nan_to_num(self._prices)
        else:
            self._standarScaler = self.set_scaler()
            self._prices = prices - prices.shift(1)
            self._prices = self._standarScaler.transform(self._prices)
            self._prices = np.nan_to_num(self._prices)
            
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state

        self.have_position = False
        self.long = False
        self.short = False
        self.open_price = 0.0
        self._offset = 0
        self.result_operation = 0
        prices = self._prices
        bars = self.bars_count
        self.account_balance = self.initial_balance
        self.account_balance_unrealized = self.initial_balance
        self.count_steps = 0
        self.rewards_steps = 0
        self.duracion = 0
        self.current_PL = 0
        
        if self.random_ofs_on_reset:
            self._offset = np.random.choice(len(prices)- bars)
        else:
            self._offset = bars
        obs = self.obs_encode(prices)
        return obs

    def step(self, action_idx):
        
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        
        
        action = Actions(action_idx)
        reward = 0.0
        done = False
        close = self._cur_close()
        trend = self.trend()
        self.result_operation = 0
        
        
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.long = True
            self.open_price = close
            #reward -= self.commission
            self.account_balance -=  self.commission
            self.count_steps += 1

            
            
        elif action == Actions.Sell and not self.have_position:
            self.have_position = True
            self.short = True
            self.open_price = close
           # reward -= self.commission
            self.account_balance -=  self.commission
            self.count_steps += 1

        
        elif action == Actions.Close and self.have_position:
            if self.long:                
                # -= self.commission
                self.account_balance -=  self.commission
                done |= self.reset_on_close
                valor_op = 12.5 * ((close - self.open_price)/0.25 )
                if not self.test:
                    #===============================================
                    if 0 < valor_op <101 and self.count_steps < 15:
                        reward+= 30
                    elif 101 < valor_op and self.count_steps < 15:
                        reward+= 50
                    elif valor_op > 0 and self.count_steps >=15:
                        reward+= 5

                else:
                    reward +=valor_op
                  
                    
                


                self.duracion = self.count_steps
                self.account_balance += 12.5 * ((close - self.open_price)/0.25 )
                self.result_operation = 12.5 * ((close - self.open_price)/0.25 ) - (self.commission * 2)
                self.have_position = False
                self.long = False
                self.open_price = 0.0
                self.count_steps = 0
                self.rewards_steps = 0
                n = 0
            
            elif self.short:
                #reward -= self.commission
                self.account_balance -=  self.commission
                done |= self.reset_on_close
                valor_op = 12.5 * ((self.open_price - close)/0.25 )
                #valor_op = ((self.open_price - close)/0.25 )
                
                if not self.test:
                    #===============================================
                    if 0 < valor_op <101 and self.count_steps < 15:
                        reward+= 30
                    elif 101 < valor_op and self.count_steps < 15:
                        reward+= 50 
                    elif valor_op > 0 and self.count_steps >=15:
                        reward+= 5
                    #==============================================
                else:
                    reward +=valor_op


                self.duracion = self.count_steps
                self.account_balance +=  12.5 * ((self.open_price - close)/0.25 )
                self.result_operation =  12.5 * ((self.open_price - close)/0.25 ) - (self.commission * 2)
                self.have_position = False
                self.short=False
                self.open_price = 0.0
                self.count_steps = 0
                self.rewards_steps = 0
                n= 0
        """
        Parte del code que se encarga de reforzar la accion <Skip> cuando hay una posicion
        buena cuando esta debajo de 20 barras y penalizar una mala en todo momento
        """
        # if action == Actions.Skip and self.have_position:  
        #     recompensa = 0
        #     if self.long:
        #         recompensa = 12.5 * ((close - self.open_price)/0.25 )
        #         if self.count_steps < 20:
        #             if recompensa >= 0 and recompensa < 101:
        #                 reward += 1
        #             elif recompensa >= 101:
        #                 reward += 2
        #             elif recompensa < -150:
        #                   reward -= 2
        #         elif self.count_steps >= 20 and recompensa <= -150:
        #             reward -= 2
            
        #     if self.short:
        #         recompensa = 12.5 * ((self.open_price - close)/0.25 )
        #         if self.count_steps < 20:                
        #             if recompensa >= 0 and recompensa < 101:
        #                 reward += 1
        #             elif recompensa >= 101:
        #                 reward += 2
        #             elif recompensa < -150:
        #                   reward -= 2
        #         elif self.count_steps >= 20 and recompensa <= -150:
        #               reward -= 2
        #         # else:
                #     reward +=0.2
        
        """
        Fraccion de codigo que premia el skip cuando hay posicion positiva, sino es 0
        """
        multiplicador = 1
        # if action == Actions.Skip and self.have_position and self.count_steps < 25:#Agregado 20/07 prueba de penalizacion en no hacer nada
        #     if self.long:
        #         recompensa = 12.5 * ((close - self.open_price)/0.25 )
        #         # if recompensa >= 101:
        #         #       reward += 0.1 * multiplicador
        #         if recompensa >= 50:
        #               reward += 0.1 * multiplicador
        #         if recompensa < -100:
        #             reward += -0.5 * multiplicador 

           
        #     if self.short:
        #         recompensa = 12.5 * ((self.open_price - close)/0.25 )
        #         # if recompensa >= 101:
        #         #       reward += 0.1 * multiplicador
        #         if recompensa >= 50:
        #               reward += 0.1 * multiplicador
        #         if recompensa < -100:
        #             reward += -0.5 * multiplicador
        
        if self.long:
            valor_op = (12.5 * ((close - self.open_price)/0.25 ))
            if action != Actions.Close and valor_op <= -150:
                reward += -2 * multiplicador
            # elif action != Actions.Close and valor_op <= -300:
            #     reward += -5 * multiplicador
            # elif action != Actions.Close and valor_op >= 100:
            #       reward += -0.8 * multiplicador
            # elif action == Actions.Skip and  self.isRising() and self.count_steps < 15:
            #         reward += 1 * multiplicador

        
        if self.short:
            valor_op = (12.5 * ((self.open_price - close)/0.25 ))
            if action != Actions.Close and valor_op <= -150:
                reward += -2 * multiplicador
            # elif action != Actions.Close and valor_op <= -300:
            #     reward += -5 * multiplicador
            # elif action != Actions.Close and valor_op >= 100:
            #         reward += -0.8 * multiplicador
            # elif action == Actions.Skip and  self.isFalling() and self.count_steps < 15:
            #         reward += 1 * multiplicador

        
        if self.have_position:
            self.count_steps +=1
            if self.long:
                self.current_PL = 12.5 * ((close - self.open_price)/0.25 )
            if self.short:
                self.current_PL = 12.5 * ((self.open_price - close)/0.25 )
        else:
            self.current_PL = 0            
        
        elliot = self.prices_witout_changes.iloc[self._offset,5]
        elliot_transformado = self._prices[self._offset,5]
        info = {
            #"instrument": self._instrument,
            "offset": self._offset,
            "position": action,
            "resultado": self.account_balance,
            "operation_result": self.result_operation,
            "operation":float(self.have_position),
            "long": float(self.long),
            "short": float(self.short),
            "current_PL": self.current_PL,
            "close_open":self.open_price,
            "close_cierre":close,
            "duracion":self.count_steps,
            "elliot":elliot,
            "elliot_transformado":elliot_transformado,
        }
        
            
        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= len(self._prices)- self.bars_count
        done |= self.account_balance < 500

        if self.have_position and not self.reward_on_close:
            if self.long:
                reward = 12.5 * ((close - self.open_price) / 0.25) - self.commission
                self.account_balance_unrealized =  (self.account_balance + 12.5 * ((close - self.open_price) / 0.25) 
                                                    - self.commission)
            
            elif self.short:
                reward = 12.5 * ((self.open_price - close) / 0.25) - self.commission
                self.account_balance_unrealized =  (self.account_balance + 12.5 * ((self.open_price - close) / 0.25)
                                                    - self.commission)
            
        obs = self.obs_encode(self._prices)
        
        
        return obs, reward, done, info
    

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
    
    def obs_encode(self,prices):
        
        if self.bars_count > 1 :
            res = np.array(prices[self._offset - (self.bars_count - 1): self._offset - (self.bars_count - 2)])
            for x in range(self._offset - (self.bars_count - 2), self._offset+ 1,1):
            
                array = np.array(prices[x: x + 1])
                res = np.append(res,array,axis=1)
        else:
            res = np.array(prices[self._offset: self._offset + 1])
            
        
        if self.have_position:
            if self.long:
                res = np.append(res,np.array(float(self.long)))
                res= np.append(res, np.array((self._cur_close() - self.open_price)))
            
            elif self.short:
                res = np.append(res,np.array( - float(self.short)))
                res= np.append(res, np.array((self.open_price - self._cur_close())))
                
            res = np.append(res,np.array(self.count_steps))
        else:
            res = np.append(res,np.array(0.0))
            res = np.append(res,np.array(0.0))
            res = np.append(res,np.array(0))
        
        # res= np.append(res,self.account_balance / self.initial_balance)
        # res= np.append(res,self.account_balance_unrealized / self.initial_balance)
    
        return res
    
    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        #open = self._prices.open[self._offset]
        #rel_close = self._prices.close[self._offset]
        # close = self._standarClose.inverse_transform(self._prices.Close_1[self._offset].reshape(1,1))
        close = self.prices_witout_changes.Close_1[self._offset]
        return close
    
    @property
    def shape(self):
        # [o, h, l, c] * bars + volume + elliot + position_flag + abs_profit + counts steps
        return 6 * self.bars_count  +2 + 1 + 1 + 1,
        # return 6 * self.bars_count,
    
    def contador_offset(self):
        return (self._offset,self.account_balance)
    
    def trend(self):
        media = self.media[self._offset]
        limit = self.limit
        if media > limit:
            return 'Alcista'
        elif media < -limit :
            return 'Bajista'
        else:
            return None
    def dump_scaler(self):        
        dump(self._standarScaler, self.name +'.bin', compress=True)
        
    def set_scaler (self):
        return load(self.name+'.bin')
    
    def isRising(self):
        up = True
        for x in range(3):
            up &= self.prices_witout_changes.iloc[self._offset - x,5] > self.prices_witout_changes.iloc[self._offset - x -1,5]
        return bool(up)
    
    def isFalling(self):
        down = True
        for x in range(3):
            down &= self.prices_witout_changes.iloc[self._offset - x,5] < self.prices_witout_changes.iloc[self._offset - x -1,5]
        return bool(down)

if __name__ == '__main__':  

    tiempoMin = 2
    resample = True
    learning_rate = 3.5e-4
    n_steps = 10
    iteraciones = 50000000
    reset_on_close = True
    time = time.time()
    # name_scalar = (str(tiempoMin)+'_'+str(resample)+'_'+str(iteraciones)+'_'+ 
    #                str(n_steps)+'_'+str(reset_on_close)+'_'+str(time))
    name_scalar = '2_True_50000000_10_True_1598705283.067478'
    path = os.path.join(os.getcwd(),'ContratoContinuo_1minFINAL.Last.txt')
    prices = load_relative_ES(path,minutos=tiempoMin,resample=resample)
    
    # log_dir = "runs/"
    # os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(StocksEnv(prices, bars_count=DEFAULT_BARS_COUNT, reset_on_close=reset_on_close, commission=DEFAULT_COMMISSION_PERC,
                      random_ofs_on_reset=True, reward_on_close=True,name_scalar=name_scalar),
                  filename=None, allow_early_resets=True)
    env.dump_scaler()
    env = DummyVecEnv([lambda: env])
    path_test = os.path.join(os.getcwd(),'06-20.txt')
    test_prices = load_relative_ES(path_test,minutos=tiempoMin,resample =resample)
    
    test_env = StocksEnv(test_prices, bars_count=DEFAULT_BARS_COUNT, reset_on_close=False,
                         commission=DEFAULT_COMMISSION_PERC,random_ofs_on_reset=True,test=True,
                         reward_on_close=True,take_scaler=True,name_scalar=name_scalar)
    test_env = DummyVecEnv([lambda: test_env])
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(test_env,verbose=1,eval_freq=1000000)
    # eval_callback = EvalCallback(test_env, callback_on_new_best=callback_on_best,
    #                              verbose=1,eval_freq=1000000)
    
    
    # model = A2C(MlpPolicy, env, verbose=1, learning_rate= 0.000001,tensorboard_log="./a2c_cartpole_tensorboard/")
    #model =A2C(MlpLstmPolicy, env, verbose=1,learning_rate=learning_rate,n_steps=n_steps,
    #           tensorboard_log="./Penalty/")
    # model =A2C(MlpPolicy, env, verbose=1,learning_rate=learning_rate,n_steps=n_steps,
    #             tensorboard_log="./Penalty/")
    # model.learn(iteraciones, callback=eval_callback)
    # model.learn(iteraciones)
    
    # env = Monitor(gym.make('CartPole-v1'),filename=None, allow_early_resets=True)
    # env = DummyVecEnv([lambda: env])
    # model =A2C(MlpPolicy, env, verbose=1)
    # model.learn(100000)
    # test_env = gym.make('CartPole-v1')
    # test_env = DummyVecEnv([lambda: test_env])
    #model.save("Reset+Old+Pena x2") 
    # del model
    
    model = A2C.load("Reset+Old+Pena x2")
    reward_list = []
    actionsList = []
    
    losFrame = []
    finalState = []
    finalInfoList = []
    finalFrame = pd.DataFrame(columns=['Steps','Final','Minimo','Maximo','Duracion Media'])
    
    
    # for w in range(10):
    #     frame = pd.DataFrame()
    #     obs = test_env.reset()
    #     info_list = []
    #     actions = []
    #     for i in range(5000):
    #         action, _states = model.predict(obs)
    #         actions.append(action)
    #         obs, rewards, dones, info = test_env.step(action)
    #         reward_list.append(rewards)
    #         info_list.append(info)
    #         frame = frame.append(info)
    #         if dones == True:
    #             break
    #     finalState.append(info[0]['resultado'])
    #     losFrame.append(frame)
    #     finalInfoList.append(info_list)
    #     actionsList.append(actions)
        
    # for idx in range(len(finalInfoList)):
    #     listaInfo = finalInfoList[idx]
    #     resultado = pd.Series(np.array(listaInfo).reshape(-1))
    #     lista = []
    #     lista = [resultado[x]['resultado'] for x in range(len(resultado))]
    #     duracion = [resultado[x]['duracion'] for x in range(len(resultado))]
    #     minimo = min(lista)
    #     maximo = max(lista)
    #     last = lista[-1]
    #     steps = len(lista)
    #     resultadoArray = pd.Series(np.array(lista).reshape(-1))
    #     duracionArray = pd.Series(np.array(duracion).reshape(-1)).mean()
    #     infoArray = {'Steps':steps,
    #                  'Final':last,
    #                  'Minimo':minimo,
    #                  'Maximo':maximo,
    #                  'Duracion Media':duracionArray}
    #     finalFrame = finalFrame.append(infoArray,ignore_index=True)
    #     resultadoArray.plot()
    #     print('Numero de lista: ',str(idx),'\n',
    #           pd.Series(np.array(actionsList[idx]).reshape(-1)).value_counts())
    #     print('------------------------------')
    
    frame = pd.DataFrame()
    obs = test_env.reset()
    info_list = []
    actions = []
    for i in range(10000):
            action, _states = model.predict(obs)
            actions.append(action)
            obs, rewards, dones, info = test_env.step(action)
            reward_list.append(rewards)
            info_list.append(info)
            frame = frame.append(info)
            if dones == True:
                break
    resultado = pd.Series(np.array(info_list).reshape(-1))
    lista = [] 
    duracion = []

    
    for x in range(len(resultado)):
        lista.append(resultado[x]['resultado'])
        duracion.append(resultado[x]['duracion'])

    resultado = pd.Series(np.array(lista).reshape(-1))
    duracionTotal = pd.Series(np.array(duracion).reshape(-1))
    print("Media de las cuentas: ",resultado[-1:])
    print("Media de las duraciones: ",duracionTotal.mean())        
    # print("Accounts values after 1 operation: \n",resultado.value_counts())
    
    
    acciones = pd.Series(np.array(actions).reshape(-1))
    print("Acciones tomadas:\n ",acciones.value_counts())    
    

    # recompensas = pd.Series(np.array(reward_list).reshape(-1))   
    # print("Recompensas recibidas: \n",recompensas.value_counts()) 
    
    
    resultado.plot()
    print("Learning Rate: ",learning_rate)
    print("N_steps: ",n_steps)
    print("Iteraciones: ",iteraciones)
    print("Resample: ",resample)
    print("Initial amount: ",INITIAL_BALANCE)
    

