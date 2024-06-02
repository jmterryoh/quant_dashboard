import streamlit as st
import pandas as pd
import yaml
import socket
import json
import hashlib
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# filepath : db_middleware.yaml
current_path = os.path.abspath(__file__)
FILEPATH_DB_MIDDLEWARE = os.path.join(os.path.dirname(current_path), 'md_client_cfg.yaml')

SECURE_KEY, HOST, PORT = None, None, None
if os.path.exists(FILEPATH_DB_MIDDLEWARE):
    with open(FILEPATH_DB_MIDDLEWARE, encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
        SECURE_KEY = _cfg['SECURE_KEY']
        SECURE_KEY = SECURE_KEY.encode('UTF-8')
        HOST = _cfg['HOST']
        PORT = _cfg['PORT']
else:
    SECURE_KEY = st.secrets["SECURE_KEY"]
    SECURE_KEY = SECURE_KEY.encode('UTF-8')
    HOST = st.secrets["HOST"]
    PORT = st.secrets["PORT"]

def encrypt_message(message, key):
    nonce = os.urandom(16)  # 16바이트(128비트)의 무작위 nonce 생성
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(message) + encryptor.finalize()
    return nonce + ciphertext

def decrypt_message(ciphertext, key):
    nonce = ciphertext[:16]  # 암호문에서 nonce 추출
    ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext

def send_data(socket, data):
    # 데이터 길이를 데이터의 제일 앞에 추가
    data_length = len(data)
    data_with_length = data_length.to_bytes(4, byteorder='big') + data

    total_sent = 0
    while total_sent < len(data_with_length):
        chunk = data_with_length[total_sent:total_sent+1024]  # 1024 바이트씩 데이터를 전송
        sent = socket.send(chunk)
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent

def receive_data(socket):
    # 먼저 데이터의 길이를 읽어옴
    data_length_bytes = socket.recv(4)
    if not data_length_bytes:
        raise RuntimeError("Socket connection broken")
    data_length = int.from_bytes(data_length_bytes, byteorder='big')

    # 데이터 길이만큼 데이터를 수신
    received_data = b''
    while len(received_data) < data_length:
        chunk = socket.recv(min(1024, data_length - len(received_data)))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        received_data += chunk

    return received_data

def fetch_result_from_remote_server(task, params):
 
    key = hashlib.sha256(SECURE_KEY).digest()[:32]
    response_data = {}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.settimeout(10)  # 타임아웃 설정
            client_socket.connect((HOST, PORT))
            print('Connected to server.')

            # Task와 Params를 JSON 형태로 구성
            task_params = {
                "task": task,
                "params": params
            }

            # Task와 Params를 서버로 전송
            encrypted_data = encrypt_message(json.dumps(task_params, ensure_ascii=False).encode('utf-8'), key)
            send_data(client_socket, encrypted_data)

            # 서버로부터 결과를 받음
            encrypted_response = receive_data(client_socket)
            response_data = json.loads(decrypt_message(encrypted_response, key).decode('utf-8'))
            #print(response_data)
            #print(f"Received result: {response_data['return']}")

        print('Disconnected from server.')
    except Exception as e:
        print(f"An error occurred: {e}")

    return response_data

def insert_stock_interest(uidx, market, code, name, pattern, description):
    task_name = "insert_stock_interest"
    params = {'uidx': uidx,
              'market': f'{market}',
              'code': f'{code}',
              'name': f'{name}',
              'pattern': f'{pattern}',
              'description':f'{description}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "error":
                #st.error(respose["return"]["data"])
                return False, respose["return"]["data"]
            else:
                return True, ""
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def update_owned_stock_transaction(uidx, market, code, name, price, quantity, trtype, trdt, reason):
    task_name = "update_owned_stock_transaction"
    params = {'uidx': uidx,
              'market': f'{market}',
              'code': f'{code}',
              'name': f'{name}',
              'price': price,
              'quantity': quantity,
              'type': f'{trtype}',
              'trdt': f'{trdt}',
              'reason':f'{reason}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "error":
                #st.error(respose["return"]["data"])
                return False, respose["return"]["data"]
            else:
                return True, ""
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def get_stocks_searched_last():
    task_name = "get_stocklist_searched_last"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    return False, "Error: no result values"
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]

    return False, "Error: no return values"

def get_algo_stocks_increase10():
    task_name = "get_algo_stocks_increase10"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    return False, "Error: no result values"
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]

    return False, "Error: no return values"

def get_algo_stock_increase10(code):
    task_name = "get_algo_stock_increase10"
    params = { 'code': f'{code}'  }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    return False, "Error: no result values"
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]

    return False, "Error: no return values"

def insert_algo_stocks_increase10(uidx, idt, i10dt, vdt, market, code, name, pattern, description):
    task_name = "insert_algo_stocks_increase10"
    params = {'uidx': uidx,
              'idt': f'{idt}',
              'i10dt': f'{i10dt}',
              'vdt': f'{vdt}',
              'market': f'{market}',
              'code': f'{code}',
              'name': f'{name}',
              'pattern': f'{pattern}',
              'description':f'{description}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                return True, ""
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def delete_algo_stocks_increase10(uidx, market, code, i10dt, description):
    task_name = "delete_algo_stocks_increase10"
    params = {'uidx': uidx,
              'market': f'{market}',
              'code': f'{code}',
              'i10dt': f'{i10dt}',
              'description':f'{description}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                return True, ""
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def insert_algo_stock_for_buy(uidx, aidx, market, code, name, pattern, stoploss_price, effective_date, allocated_amount, algorithm_buy, algorithm_sell, description):
    task_name = "insert_algo_stock_for_buy"
    params = {'uidx': int(uidx),
              'aidx': int(aidx),
              'market': f"{market}",
              'code': f"{code}",
              'name': f"{name}",
              'pattern': f"{pattern}",
              'stoploss_price' : int(stoploss_price),
              'effective_date': f"{effective_date}",
              'allocated_amount': allocated_amount,
              'algorithm_buy': f"{algorithm_buy}",
              'algorithm_sell': f"{algorithm_sell}",
              'description': f"{description}" }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                return True, ""
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def update_algo_stock_for_sell(market, code, pattern, algorithm_sell, stoploss_price, description):
    task_name = "update_algo_stock_for_sell"
    params = {'market': f'{market}',
              'code': f'{code}',
              'pattern':f'{pattern}',
              'algorithm_sell':f'{algorithm_sell}',
              'stoploss_price':int(stoploss_price),
              'description': f"{description}"}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                return True, ""
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def is_holiday(string_date):

    task_name = "calendar_holiday_check"
    params = {'date': f'{string_date}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, ""
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def get_calendar_holidays(string_date_from):

    task_name = "get_calendar_holidays"
    params = {'date_from': f'{string_date_from}'}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    return False, "Error: no result values"
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]
        else:
            return False, "Error: no result values"
    else:
        return False, "Error: no return values"

def get_users_account_idx():
    task_name = "get_users_account_idx"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    return False, "Error: no result values"
            elif respose["return"]["result"] == "error":
                return False, respose["return"]["data"]

    return False, "Error: no return values"

def get_stocks_interest():
    task_name = "get_stocklist_interest"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return df["code"].tolist()
                else:
                    print("get_stocks_interest: None")
    return None

def get_stocks_owned():
    task_name = "get_stocks_owned_all"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if not df.empty:                
                    df = df[df['quantity'] > 0]
                    if df.empty:
                        return None
                    return df["code"].tolist()
    return None

def get_algo_stocklist_for_buy():
    task_name = "get_algo_stocklist_for_buy"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return df["code"].tolist(), dict(zip(df["code"], df["market"]))
                else:
                    print("get_algo_stocklist_for_buy: None")
    return None, None

def get_algo_stock_for_buy_trade_exists(uidx, aidx, code, effective_date, algorithm_buy, description):
    task_name = "get_algo_stock_for_buy_trade_exists"
    params = { 
        "uidx": int(uidx),
        "aidx": int(aidx),
        "code": f"{code}",
        "effective_date": f"{effective_date}",
        "algorithm_buy": f"{algorithm_buy}",
        "description": f"{description}"
    }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    print("get_algo_stock_for_buy_trade_exists: None")
    return False, None

def get_algo_stock_for_sell_trade_exists(uidx, aidx, code, algorithm_buy):
    task_name = "get_algo_stock_for_sell_trade_exists"
    params = { 
        "uidx": int(uidx),
        "aidx": int(aidx),
        "code": f"{code}",
        "algorithm_buy": f"{algorithm_buy}"
    }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    print("get_algo_stock_for_sell_trade_exists: None")
    return False, None

def get_algo_stocklist_for_sell():
    task_name = "get_algo_stocklist_for_sell"
    params = {}
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return df["code"].tolist(), dict(zip(df["code"], df["market"]))
                else:
                    print("get_algo_stocklist_for_sell: None")
    return None, None

def get_algo_stocks_for_buy(aidx):
    task_name = "get_algo_stocklist_for_buy"
    params = { 'aidx': aidx }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    print("get_algo_stocks_for_buy: None")
    return False, None

def get_algo_stocks_for_sell(aidx):
    task_name = "get_algo_stocklist_for_sell"
    params = { 'aidx': aidx }
    respose = fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
                if len(df) > 0:
                    return True, df
                else:
                    print("get_algo_stocks_for_sell: None")
    return False, None


#def fetch_threaded_result(task, params):
#    result = None
#    thread = threading.Thread(target=lambda: fetch_result_from_remote_server(task, params))
#    thread.start()
#    thread.join()
#    return result

#task_name = "get_stocklist_searched"
#params = {'idt': '20240323', 'seq': 1}
#fetch_result_from_remote_server(task_name, params)
