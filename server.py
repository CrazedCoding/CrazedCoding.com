#WebSocket Server
import os
from os import listdir
from os.path import isfile, join
import asyncio
import time
import random
import websockets
import posixpath
import mimetypes
import base64
from http import HTTPStatus
from messages_pb2 import Message, Algorithm
import google.protobuf.json_format as json_format
import threading
import ssl
from http.server import HTTPServer, BaseHTTPRequestHandler
import math, datetime, time

import io
from urllib.request import urlopen, Request
#Email
from urllib.parse import urlparse
import re
import sys
from records import Records

ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
www_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"ssl")

crt_path = os.path.realpath("/etc/letsencrypt/live/crazedcoding.com/fullchain.pem")
key_path = os.path.realpath("/etc/letsencrypt/live/crazedcoding.com/privkey.pem")



loop = asyncio.get_event_loop()
BYTE_RANGE_RE = re.compile(r'bytes=(\d+)-(\d+)?$')

records = Records(loop)

def parse_byte_range(byte_range):
    '''Returns the two numbers in 'bytes=123-456' or throws ValueError.
    The last number or both numbers may be None.
    '''
    if byte_range.strip() == '':
        return None, None

    m = BYTE_RANGE_RE.match(byte_range)
    if not m:
        raise ValueError('Invalid byte range %s' % byte_range)

    first, last = [x and int(x) for x in m.groups()]
    if last and last < first:
        raise ValueError('Invalid byte range %s' % byte_range)
    return first, last

class WebSocketServerProtocolWithHTTP(websockets.WebSocketServerProtocol):
    """Implements a simple static file server for WebSocketServer"""

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_media(self, www_path, algorithms_root, short_path, request_headers, response_headers, ctype, parsed_url):
        
        dangerous_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), short_path)
        # Validate the path
        if os.path.commonpath((algorithms_root, dangerous_path)) != algorithms_root or \
                not os.path.exists(dangerous_path) or not os.path.isfile(dangerous_path):
            print("404 NOT FOUND")
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'
        try:
            user = short_path.split('/')[1]
            query = parsed_url.query
            split_query = query.split('=')
            if len(split_query) != 2 or split_query[0] != "hash":
                print("404 NOT FOUND")
                return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'
            elif not records.check_http_request_auth(user, split_query[1]):
                print("401 UNAUTHORIZED")
                return HTTPStatus.UNAUTHORIZED, [], b'401 NOT AUTHORIZED'
        except:
            print("404 NOT FOUND")
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'


        try:
            f = open(dangerous_path, 'rb')
            fs = os.fstat(f.fileno())
            file_len = fs[6]

            try:
                self.range = parse_byte_range(request_headers['Range'])
            except:
                self.range = (0, min(1024, file_len))
            first, last = self.range
            
            if first >= file_len:
                print("404 NOT FOUND")
                return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'

            #response_headers.append(('Content-type', ctype))
            response_headers.append(('Accept-Ranges', 'bytes'))

            if last is None or last >= file_len:
                last = file_len - 1
            response_length = last - first + 1
            response_headers.append(('Content-Length', str(response_length)))

            response_headers.append(('Content-Range',
                            'bytes %s-%s/%s' % (first, last, file_len)))
            #response_headers.append(('Last-Modified', self.date_time_string(fs.st_mtime)))

            f.seek(first)
            body = f.read(response_length)
            f.close()

            return HTTPStatus.PARTIAL_CONTENT, response_headers, body
        except IOError:
            print("404 NOT FOUND")
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'


    async def process_request(self, path, request_headers):
        """Serves a file when doing a GET request with a valid path"""
        self.max_size = 2**20

        if "Upgrade" in request_headers:
            return  # Probably a WebSocket connection
        parsed = urlparse(path)
        query_params = parsed.query.split("=")
        new_query_params = []
        for query in query_params:
            split_query = query.split("&")
            for element in split_query:
                new_query_params.append(element)
        query_params = new_query_params
        path = parsed.path

        # print(request_headers)

        if path == '/' or path == '':
            path = '/index.html'

        response_headers = [
            ('Server', 'CrazedCoding.com')
        ]
        server_root = os.path.dirname(os.path.abspath(__file__))
        www_root = os.path.join(server_root,"www")
        algorithms_root = os.path.join(server_root,"algorithms")
        short_path = path[1:]
        www_path = os.path.realpath(os.path.join(www_root, short_path))

        print("GET", path, end='\n')
        # print(os.path.commonpath((www_root, www_path)) != www_root)
        # print(os.path.commonpath((algorithms_root, os.path.realpath(os.path.join(server_root,short_path)))), algorithms_root)
        # Validate the path
        if os.path.commonpath((www_root, www_path)) != www_root and \
                os.path.commonpath((algorithms_root, os.path.realpath(os.path.join(server_root,short_path)))) != algorithms_root or \
                len(short_path.split("..")) > 1:
            print("404 NOT FOUND", path, end='\n')
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'
        ctype = self.guess_type(path)
        if "audio" in ctype or "video" in ctype:
            return self.send_media(www_path, algorithms_root, short_path, request_headers, response_headers, ctype, parsed)
        else:
            template = __import__("template")
            return template.render(server_root, query_params, www_root, www_path, algorithms_root, short_path, request_headers, response_headers, ctype, parsed)

    def guess_type(self, path):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """
 
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']
 
    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        })

def process_message(websocket, proto, serialized_proto):
    if proto.type == Message.CAPTCHA:
        records.send_captcha(websocket)
    elif proto.type == Message.REGISTER and records.check_captcha(websocket, proto):
        result_message = Message()
        
        if type(re.match(r'^[a-zA-Z]+$', proto.auth.user)) == type(None) or len(proto.auth.user) < 3:
            result_message.type = Message.ERROR
            result_message.message = "User name must be at least three (3) characters and consist only of letters!"
            result_message.details = "Please type a new name, and try again."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            records.send_captcha(websocket)
            return

        user_by_name = records.get_user_by_name(proto.auth.user)
        user_by_email = records.get_user_by_email(proto.auth.email)

        user = None

        if type(user_by_name) != type(None):
            result_message.type = Message.ERROR
            result_message.message = "User name already taken!"
            result_message.details = "Please choose another name, or, if this is your account, reset your password by clicking the \"Reset Password\" button."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            records.send_captcha(websocket)
            return

        if type(user_by_name) != type(None):
            user = user_by_name
        if type(user_by_email) != type(None):
            user = user_by_email


        if type(user) != type(None) and not user.auth.validated:
            date = time.mktime(datetime.datetime.now().timetuple())
            last_captcha_date = user.captcha.date/1000.0
            delta_time = date-last_captcha_date
            if delta_time < records.captcha_timeout:
                result_message.type = Message.ERROR
                result_message.message = "Validation email already sent. Please remember to check your \"Spam\" folder!"
                result_message.details = "Expires "+str(int(math.floor(records.captcha_timeout-delta_time)))+" seconds from now. Please wait and try again."
                asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                records.send_captcha(websocket)
                return
            else:
                name = user.auth.user
                email = user.auth.email
                captcha_message = records.send_validation(websocket, name, email)
                if type(captcha_message) == type(None):
                    return
                user.captcha.key = captcha_message.captcha.key
                user.captcha.image = captcha_message.captcha.image
                user.captcha.date = captcha_message.captcha.date
                records.write_user(user)
                result_message.type = Message.PROGRESS
                result_message.message = "Validation email re-sent. Please remember to check your \"Spam\" folder!"
                result_message.details = "You have "+str(records.captcha_timeout)+" seconds to respond."
                asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                records.send_captcha(websocket)
                return
        
        if type(user_by_email) != type(None):
            result_message.type = Message.ERROR
            result_message.message = "Email already registered!"
            result_message.details = "Please reset your password by clicking the \"Reset Password\" button, or choose another email."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            records.send_captcha(websocket)
            return
             
        captcha_message = records.send_validation(websocket, proto.auth.user, proto.auth.email)
        if type(captcha_message) == type(None):
            return
        proto.captcha.key = captcha_message.captcha.key
        proto.captcha.image = captcha_message.captcha.image
        proto.captcha.date = captcha_message.captcha.date
        records.write_user(proto)
        result_message.type = Message.PROGRESS
        result_message.message = "Registration email sent."
        result_message.details = "You have "+str(records.captcha_timeout)+" seconds to respond."
        asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
        records.send_captcha(websocket)
        return
    elif proto.type == Message.REQUEST_PASSWORD_RESET and records.check_captcha(websocket, proto):
        result_message = Message()
        
        user_by_name = records.get_user_by_name(proto.auth.user)
        user_by_email = records.get_user_by_email(proto.auth.email)

        user = None

        if type(user_by_name) != type(None):
            user = user_by_name
        if type(user_by_email) != type(None):
            user = user_by_email

        if type(user) == type(None):
            result_message.type = Message.ERROR
            result_message.message = "No user registered by that name or email!"
            result_message.details = "Please enter a valid email or user name."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            records.send_captcha(websocket)
            return
        else:
            date = time.mktime(datetime.datetime.now().timetuple())
            last_captcha_date = user.captcha.date/1000.0
            delta_time = date-last_captcha_date
            if delta_time < records.captcha_timeout:
                result_message.type = Message.ERROR
                result_message.message = "Validation email already sent. Please remember to check your \"Spam\" folder!"
                result_message.details = "Expires "+str(int(math.floor(records.captcha_timeout-delta_time)))+" seconds from now. Please wait and try again."
                asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                records.send_captcha(websocket)
                return
            else:
                name = user.auth.user
                email = user.auth.email
                captcha_message = records.send_validation(websocket, name, email)
                if type(captcha_message) == type(None):
                    return
                user.captcha.key = captcha_message.captcha.key
                user.captcha.image = captcha_message.captcha.image
                user.captcha.date = captcha_message.captcha.date
                records.write_user(user)
                result_message.type = Message.REQUEST_PASSWORD_RESET
                result_message.message = "Validation email re-sent. Please remember to check your \"Spam\" folder!"
                result_message.details = "You have "+str(records.captcha_timeout)+" seconds to respond."
                asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                records.send_captcha(websocket)
                return
    elif proto.type == Message.VALIDATE:
        result_message = Message()
        users = records.read_users()
        for user in users:
            if user.auth.user.lower() == proto.auth.user.lower():
                if user.captcha.key == proto.captcha.key:
                    user.captcha.key = ""
                    user.captcha.image = b''
                    user.captcha.date = 0
                    user.auth.validated = True
                    records.write_user(user)
                    websocket.user = user
                    records.send_captcha(websocket)
                    result_message.type = Message.SET_PASSWORD
                    result_message.message = "Email validated!"
                    result_message.details = "You may now set your password!"
                    asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                    websocket.can_set_password = True
                    return
                elif user.auth.validated:
                    result_message.type = Message.ERROR
                    result_message.message = "Email already validated."
                    result_message.details = "Please use the password reset functionality to change your password."
                    asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
                    asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
                    return

        result_message.type = Message.ERROR
        result_message.message = "User not registered."
        result_message.details = "Please use the \"Sign Up\" functionality to create an account."
        asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
        asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
        return
    elif proto.type == Message.SET_PASSWORD and records.check_captcha(websocket, proto):
        result_message = Message()

        user = records.get_user_by_name(proto.auth.user)

        if type(user) == type(None) or not user.auth.validated:
            result_message.type = Message.ERROR
            result_message.message = "User does not exist!"
            result_message.details = "Please sign up with a valid email and user name."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return
        if not hasattr(websocket, 'user') or type(websocket.user) == type(None) or not hasattr(websocket, 'can_set_password') or not websocket.can_set_password:
            result_message.type = Message.ERROR
            result_message.message = "Not authorized!"
            result_message.details = "Please sign up with a valid email and user name."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return
        if websocket.user.auth.user != proto.auth.user:
            result_message.type = Message.ERROR
            result_message.message = "Not your account!"
            result_message.details = "This incident will be reported."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return
        if type(re.search(r'(?=.*\d)(?=.*[a-zA-Z]).{6,}', proto.auth.password)) == type(None):
            result_message.type = Message.ERROR
            result_message.message = "Invalid password!"
            result_message.details = "Please enter a valid password. Use at least one number, at least one letter, and at least six characters."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return

        websocket.can_set_password = False
        records.delete_user(user)
        user.auth.password = proto.auth.password
        websocket.user = user
        key = str(math.floor(random.randrange(1E6, 1E7)))
        websocket.user.auth.hash = key

        records.write_user(websocket.user)

        result_message.type = Message.LOGIN
        result_message.auth.user = websocket.user.auth.user
        result_message.auth.hash = websocket.user.auth.hash
        result_message = records.censor_user(result_message)
        result_message.auth.email = websocket.user.auth.email

        asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)

        result_message = Message()
        result_message.type = Message.PROGRESS
        result_message.message = "Password set!"
        result_message.details = "Bonus: you're also now logged in. Press the \"Log Out\" button or close the tab to log out."
        asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)

        records.send_captcha(websocket)
        records.send_websocket_auth(websocket, result_message)
        return
    elif proto.type == Message.SAVE_ALGORITHM and records.check_websocket_auth(websocket, proto.auth.hash, True):
        records.save_algorithm(websocket, proto)
        return
    elif proto.type == Message.DELETE_ACCOUNT and records.check_websocket_auth(websocket, proto.auth.hash, True):

        users_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"users")

        backup_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"deleted")
        if not os.path.exists(backup_root):
            os.mkdir(backup_root)

        deleted_user_folder = os.path.join(backup_root, proto.auth.user)
        if not os.path.exists(deleted_user_folder):
            os.mkdir(deleted_user_folder)

        deleted_videos_folder = os.path.join(deleted_user_folder, "videos")
        if not os.path.exists(deleted_videos_folder):
            os.mkdir(deleted_videos_folder)
        
        videos_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"videos")
        if not os.path.exists(videos_root):
            os.mkdir(videos_root)
            
        user_videos_folder = os.path.join(videos_root, proto.auth.user)
        if not os.path.exists(user_videos_folder):
            os.mkdir(user_videos_folder)

        deleted_video_folder = os.path.join(deleted_user_folder,"videos")
        if not os.path.exists(deleted_video_folder):
            os.mkdir(deleted_video_folder)

        user_videos_folder = os.path.join(videos_root, proto.auth.user)
        if not os.path.exists(user_videos_folder):
            os.mkdir(user_videos_folder)

        user_files = [f for f in listdir(user_videos_folder) if isfile(join(user_videos_folder, f))]

        for i in range(len(user_files)):
            try:
                old_file = os.path.join(user_videos_folder, user_files[i])
                new_file = os.path.join(deleted_videos_folder, user_files[i])
                os.rename(old_file, new_file)
            except Exception as e: 
                print(str(e))
                message = Message()
                message.type = Message.ERROR
                message.message = "Error while deleting video!"
                message.details = "Please contact the site administrators to resolve this issue."
                asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
                return
        try:
            current_proto = records.get_user_by_name(proto.auth.user)
            user_file_path = "{}{}".format(current_proto.auth.email.lower(), ".proto")
            old_proto = os.path.join(users_root, user_file_path)
            new_proto = os.path.join(deleted_user_folder, user_file_path)
            os.rename(old_proto, new_proto)
        except Exception as e: 
            print(str(e))
            message = Message()
            message.type = Message.ERROR
            message.message = "Error while deleting account!"
            message.details = "Account videos deleted, but the user was not. Please contact the site administrators to resolve this issue."
            asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
            return

        message = Message()
        message.type = Message.DELETE_ACCOUNT
        message.message = "Account successfully deleted."
        message.details = "Your account no longer exists."
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)

    elif proto.type == Message.LOGIN and records.check_captcha(websocket, proto):
        result_message = Message()
        
        user_by_name = records.get_user_by_name(proto.auth.user)
        user_by_email = records.get_user_by_email(proto.auth.email)

        user = None

        if type(user_by_name) != type(None):
            user = user_by_name
        if type(user_by_email) != type(None):
            user = user_by_email

        security_risk = type(user) == type(None) or \
        type(user.auth) == type(None) or \
        type(proto.auth) == type(None) or \
        type(proto.auth.user) == type(None) or \
        user.auth.password != proto.auth.password or \
        len(proto.auth.user) < 3


        if security_risk:
            result_message.type = Message.ERROR
            result_message.message = "Invalid name, email, or password!"
            result_message.details = "Please enter a valid name, email, and password."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            records.send_captcha(websocket)
            return

        if user.auth.password == "":
            result_message.type = Message.ERROR
            result_message.message = "This account needs to set a password!"
            result_message.details = "Use the \"Reset Password\" functionality to get a password-reset email."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return

        if type(user) != type(None) and not user.auth.validated:
            result_message.type = Message.ERROR
            result_message.message = "This account needs to validate it's email!"
            result_message.details = "Please remember to check your \"Spam\" folder! Use the \"Reset Password\" functionality to get a new validation email."
            asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
            asyncio.run_coroutine_threadsafe(websocket.close(), loop=loop)
            return
        
        records.delete_user(user)
        websocket.user = user
        key = str(math.floor(random.randrange(1E6, 1E7)))
        websocket.user.auth.hash = key

        records.write_user(websocket.user)

        result_message.type = Message.LOGIN
        result_message.auth.user = websocket.user.auth.user
        result_message.auth.hash = websocket.user.auth.hash
        result_message = records.censor_user(result_message)
        result_message.auth.email = websocket.user.auth.email

        asyncio.run_coroutine_threadsafe(websocket.send(result_message.SerializeToString()), loop=loop)
        return

async def on_connection(websocket, path):
    records.new_connection(websocket)
    try:
        while True:
            #now = datetime.datetime.utcnow().isoformat() + 'Z'
            #await websocket.send(now)
            length = int(await websocket.recv())
            update = int(await websocket.recv())
            array = b''
            last_progress = None
            while len(array) != length:
                array += await websocket.recv()
                if update != 0:
                    progress = math.floor(len(array)*100.0/length)
                    if progress != last_progress:
                        message = Message()
                        message.type = Message.HALT
                        message.message = "Uploading..."
                        message.details = "{}%".format(progress)
                        await websocket.send(message.SerializeToString())
                        last_progress = progress
            message = Message()
            threading.Thread(target=process_message, args=(websocket, message.FromString(array), array)).start()
            del message
            del array
    except:
        records.destroy_connection(websocket)
        pass

def redirect():
    class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(301)
            self.send_header("Location", "https://www.CrazedCoding.com"+self.path)
            self.end_headers()
    httpd = HTTPServer(('', 80), SimpleHTTPRequestHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    users_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"users")
    if not os.path.exists(users_root):
        os.mkdir(users_root)

    threading.Thread(target=redirect).start()

    ssl_context.load_cert_chain(crt_path, key_path)
    
    start_server = websockets.serve(on_connection, '', 443,ssl=ssl_context,
                                    create_protocol=WebSocketServerProtocolWithHTTP)

    print("Running server at https://CrazedCoding.com:443/")

    loop.run_until_complete(start_server)
    loop.run_forever()
    