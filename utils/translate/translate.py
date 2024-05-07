import requests
import random
from hashlib import md5
 
 
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()
 
 
class BaiduAPI:
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
 
    def __init__(self):
        self._appid = "xxx"  ## change with your own appid 
        self._appkey = "xxxx" # change with your own appkey
 
    @property
    def appid(self):
        return self._appid
 
    @appid.setter
    def appid(self, app_id):
        self._appid = app_id
 
    @property
    def appkey(self):
        return self._appkey
 
    @appkey.setter
    def appkey(self, app_key):
        self._appkey = app_key
 
    def translate(self, text, from_lang='auto', to_lang='zh'):
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + text + str(salt) + self.appkey)
 
        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
 
        # Send request
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()
 
        # Show response
        # print(json.dumps(result, indent=4, ensure_ascii=False))
        return result["trans_result"][0]["dst"]


# from Baidu_Text_transAPI import BaiduAPI
if __name__ == "__main__":
    baidu_api = BaiduAPI()
    text = "hello"
    print(baidu_api.translate(text))  # 剩余两个参数可以使用默认值，也可以指定

    
