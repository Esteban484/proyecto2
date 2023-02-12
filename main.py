from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import joblib
import sklearn



global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('static/shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='templates/')
picFolder = os.path.join('static','shots')

app.config['UPLOAD_FOLDER'] = picFolder

camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame


def imagen():
    img1=cv2.imread("static/shots/img_1.jpg")
    arr=np.asarray(img1)
    return arr


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                cont=1
                now = datetime.datetime.now()
                p = os.path.sep.join(['static/shots', "img_{}.jpg".format(cont)])
                cv2.imwrite(p, frame)
                
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
#Caracteristicas de la imagen
img=cv2.cvtColor(imagen(),cv2.COLOR_BGR2GRAY)



def hu_mom (img):
# Calculamos los momentos de Hu de cada imagen
    moments = cv2.HuMoments(cv2.moments(img)).flatten()
    return moments

def hog_mom(img):
    # imagen sea de tipo CV_8U
    Xtr = cv2.convertScaleAbs(imagen())
    # Inicializamos la lista de momentos de HOG
    # Calculamos los momentos de HOG de cada imagen
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(img)
    return hog_features



#Modelos
modelSVM_Hu= joblib.load('modelos/ModeloSVM_Hu.joblib')
im = cv2.imread("static/shots/img_1.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
new_image = cv2.HuMoments(cv2.moments(im)).flatten()
ne = np.array([new_image])
prediction = modelSVM_Hu.predict(ne)

print("La nueva imagen es un {}".format(prediction[0]))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/request',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Tomar foto':
                global capture
                capture=1
                imageList = os.listdir('static/shots')
                imagelist = ['shots/' + image for image in imageList]
                pred=prediction[0]
                return render_template('index.html',imagelist=imagelist,pred=pred)
            
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Apagar/Encender':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Iniciar/Detener Grabacion':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          

@app.route("/")
def index():
        imageList = os.listdir('static/shots')
        imagelist = ['shots/' + image for image in imageList]
        return render_template('index.html',imagelist=imagelist)
        
@app.route("/results",methods=['POST',"GET"])
def result ():
        imageList = os.listdir('static/shots')
        imagelist = ['shots/' + image for image in imageList]
        hu=hu_mom(img)
        hog=hog_mom(img)
        pred=prediction[0]
        
        return render_template('index.html',imagelist=imagelist,hu=hu,hog=hog,pred=pred)

    
if __name__ == '__main__':
        app.run()