from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import argparse
import json
import keras_gaia.project as project_utils

parser = argparse.ArgumentParser(description='Use a neural network project to predict data')
parser.add_argument('--base', help='base path for relative pathes in project file')
parser.add_argument('--port', help='port for the HTTP server')
parser.add_argument('projectFile', help='project file')

args = parser.parse_args()

project = project_utils.load_json(args.projectFile, {
    'base': args.base,
    'loadWeights': True
})

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['content-length'])
        input_content = self.rfile.read(content_length)
        input_data = json.loads(input_content, 'UTF-8')

        output_data = project.predict(input_data)
        output_content = json.dumps(output_data, sort_keys=True)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(output_content)

        return

try:
    server = HTTPServer(('', int(args.port)), RequestHandler)
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()
