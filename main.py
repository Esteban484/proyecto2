from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import joblib
import sklearn
from PIL import Image,ImageFilter
import matplotlib as plt
import dlib
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



def imagen():
    img1=cv2.imread("static/shots/img_1.jpg")
    arr=np.asarray(img1)
    return arr


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture==1):
                capture=0
                cont=1
                now = datetime.datetime.now()
                p = os.path.sep.join(['static/shots', "img_{}.jpg".format(cont)])
                cv2.imwrite(p, frame)
            if(capture==2):
                capture=0
                cont=2
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
    images=np.array(Image.open(img))
# Calculamos los momentos de Hu de cada imagen
    gray=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    return moments

def hog_mom(img):
    im = Image.open(img)
    im = im.convert('L')
    im_resized = im.resize((200, 200))
    im=np.array(im_resized)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(im)
    return hog_features[1:10]

def detector2(img3):
    #prediction=0
    img = cv2.imread(img3)
    # Inicializar detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()

    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en imagen
    faces = detector(gray, 1)
    print(faces)
   

    modelSVM_Hu= joblib.load('modelos/ModeloSVM_Hu.joblib')
    im=np.array(img)
    gray2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hu_features = cv2.HuMoments(cv2.moments(gray2)).flatten()
    ne = np.array([hu_features])
    prediction2 = int(modelSVM_Hu.predict(ne))
    print("MOMENTOS DE HU")
    print(hu_mom(img3))

    print("MOMENTOS DE HOG")
    print(hog_mom(img3))
    return prediction2


def detector(img2):
    #prediction=0
    img = cv2.imread(img2)
    # Inicializar detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()

    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en imagen
    faces = detector(gray, 1)
    print(faces)
    prediction=None
    #print(prediction)
    

    if len(faces) == 0:
        prediction=1
        
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            a=img[y:y+h, x:x+w]
        modelSVM_Hog= joblib.load('modelos/ModeloRF_Hog.joblib')
        im=np.array(a)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(im)
        ne = np.array([hog_features[0:1000]])
        prediction = int(modelSVM_Hog.predict(ne))

        print("MOMENTOS DE HU")
        print(hu_mom("static/shots/img_1.jpg"))

        print("MOMENTOS DE HOG")
        print(hog_mom("static/shots/img_1.jpg"))
    return prediction
        
        
  
    

#Modelos


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
                pred=detector("static/shots/img_1.jpg")
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
                          
@app.route('/results',methods=['POST','GET'])
def tasks2():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click2') == 'Capturar foto':
                global capture
                capture=2
                imageList = os.listdir('static/shots')
                imagelist = ['shots/' + image for image in imageList]
                pred2=detector2("static/shots/img_2.jpg")
                return render_template('rdn.html',imagelist=imagelist,pred2=pred2)
            
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
        pred=detector("static/shots/img_1.jpg")
        pred2=detector2("static/shots/img_1.jpg")
        
        return render_template('index.html',imagelist=imagelist,hu=hu,hog=hog,pred=pred,pred2=pred2)

@app.route("/rdn",methods=['POST',"GET"])
def rdn ():
        imageList = os.listdir('static/shots')
        imagelist = ['shots/' + image for image in imageList]
        
        
        return render_template('rdn.html',imagelist=imagelist)


    
if __name__ == '__main__':
        app.run()