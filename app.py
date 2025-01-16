from flask import  Flask,render_template,request
import cv2
import mediapipe as mp
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy
import os
from cvzone.PoseModule import PoseDetector
import cvzone
import webbrowser 


app=Flask(__name__)
# Functions
def calculate_focal():
    cap=cv2.VideoCapture(0)
    detector=FaceMeshDetector(maxFaces=1)
    i=0
    f=[]
    while i<90:
        sucess,img=cap.read()
        img,faces=detector.findFaceMesh(img)
        if faces:
            face=faces[0]
            left_pupil=face[145]
            right_pupil=face[374]
            cv2.line(img,left_pupil,right_pupil,(0,200,0),3)
            cv2.circle(img,left_pupil,5,(255,0,255),cv2.FILLED)
            cv2.circle(img,right_pupil,5,(255,0,255),cv2.FILLED)
            w,_=detector.findDistance(left_pupil,right_pupil)
            W=6.3
            d=50
            f.append((w*d)/W)
            cv2.imshow("Image",img)
            cv2.waitKey(1)
            i+=1
    return sum(f)/90


#Function End
# # #Database
# import mysql.connector
# db=mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='sanket@123',
#     database='user'
# )
# mycursor=db.cursor()
# sql="INSERT INTO user_register(f_name,l_name,email,weight,age,focal,gender,password) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
# val=('Sanket','Gadhe','pratik@gmail.com','52','21','508','male','sanket@123')
# sql = "UPDATE user_register SET f_name=%s, l_name=%s, weight=%s, age=%s, focal=%s, gender=%s, password=%s WHERE email=%s"
# # val = ('Sanket', 'Gadhe', '52', '21', '508', 'male', 'sanket@123', 'sanket366@gmail.com')
# mycursor.execute(sql,val)
# db.commit()


#End of Database
@app.route("/")
@app.route("/home")
def home():
    return render_template('Template/login.html')
# @app.route('/index')
# def index():
#     return render_template('index.html')
# @app.route('/focal',methods=['POST','GET'])
# def focal():
#     # focal=calculate_focal()
#     focal=508
#     output=request.form.to_dict()
#     fname=output['fname']
#     lname=output['lname']
#     email=output['email']
#     weight=output['weight']
#     age=output['age']
#     gender=output['gender']
#     password=output['password']
#     sql = "UPDATE user_register SET f_name=%s, l_name=%s, weight=%s, age=%s, focal=%s, gender=%s, password=%s WHERE email=%s"
#     val = (fname, lname, weight, age, focal, gender, password, email)
#     mycursor.execute(sql,val)
#     db.commit()
#     return render_template('index.html',fname=fname,lname=lname,email=email,age=age,weight=weight,gender=gender,focal=focal)
@app.route('/measurement',methods=['POST','GET'])
def measurement():
    f=508
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap=cv2.VideoCapture(0)
    detector=FaceMeshDetector()
    i=0
    avg_shoulder=[]
    avg_waist=[]
    avg_shirt=[]
    avg_hand=[]
    avg_ankle=[]
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while i<180:
            success,img=cap.read()
            img,faces=detector.findFaceMesh(img,draw=False)
            if faces:
                face=faces[0]
                left_pupil=face[145]
                right_pupil=face[374]
                results = pose.process(img)
                if results.pose_landmarks:
                    landmarks = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in results.pose_landmarks.landmark]
                left_shoulder = (landmarks[11][0]+7,landmarks[11][1]-15)
                right_shoulder = (landmarks[12][0]-7,landmarks[12][1]-15)
                left_waist=(landmarks[23][0]+25,landmarks[23][1]-40)
                right_waist=(landmarks[24][0]-25,landmarks[24][1]-40)
                top_shoulder = (landmarks[12][0]-7,landmarks[12][1]-10)
                bottom_waist=(landmarks[24][0]-25,landmarks[24][1]+30)
                right_hand=(landmarks[15][0],landmarks[15][1])
                left_hand=(landmarks[16][0],landmarks[16][1])
                right_ankle = (landmarks[28][0],landmarks[28][1])
                # Shoulder
                cv2.line(img,left_shoulder,right_shoulder,(0,200,0),3)
                cv2.circle(img,left_shoulder,3,(25,0,255),cv2.FILLED)
                cv2.circle(img,right_shoulder,3,(25,0,255),cv2.FILLED)
                # Waist
                cv2.line(img,left_waist,right_waist,(0,200,0),3)
                cv2.circle(img,left_waist,3,(255,0,25),cv2.FILLED)
                cv2.circle(img,right_waist,3,(255,0,25),cv2.FILLED)
                # TO_bottom
                cv2.line(img,top_shoulder,bottom_waist,(0,200,0),3)
                cv2.circle(img,bottom_waist,3,(255,0,25),cv2.FILLED)
                cv2.circle(img,top_shoulder,3,(255,0,25),cv2.FILLED)
                # TO_hand
                cv2.line(img,top_shoulder,left_hand,(0,200,100),3)
                cv2.circle(img,left_hand,3,(255,98,25),cv2.FILLED)
                cv2.circle(img,top_shoulder,3,(255,98,25),cv2.FILLED)
                # TO_ANKLE
                cv2.line(img,right_waist,right_ankle,(100,200,100),3)
                cv2.circle(img,right_waist,3,(255,198,25),cv2.FILLED)
                cv2.circle(img,right_ankle,3,(255,98,25),cv2.FILLED)
                # Pupil
                cv2.line(img,left_pupil,right_pupil,(0,200,0),3)
                cv2.circle(img,left_pupil,5,(255,0,255),cv2.FILLED)
                cv2.circle(img,right_pupil,5,(255,0,255),cv2.FILLED)
                w,_=detector.findDistance(left_pupil,right_pupil)
                W=6.3
                distance=(W*f)/w
                waist_d,_=detector.findDistance(left_waist,right_waist)
                shoulder_d,_=detector.findDistance(left_shoulder,right_shoulder)
                shirt_d,_=detector.findDistance(top_shoulder,bottom_waist)
                hand_d,_=detector.findDistance(top_shoulder,left_hand)
                ankle_d,_=detector.findDistance(right_ankle,right_waist)
                shoulder_Org=(shoulder_d*distance)/f
                waist_Org=(waist_d*distance)/f
                shirt_Org=(shirt_d*distance)/f
                hand_Org=(hand_d*distance)/f
                ankle_Org=(ankle_d*distance)/f
                avg_shirt.append(shirt_Org/100)
                avg_shoulder.append(shoulder_Org/100)
                avg_waist.append(waist_Org/100)
                avg_hand.append(hand_Org/100)
                avg_ankle.append(ankle_Org/100)
                cvzone.putTextRect(img,f'Depth:{int(distance)}cm',(10,50),scale=2)
                cvzone.putTextRect(img,f'Shoulder:{int(shoulder_Org)}cm',(10,150),scale=2)
                cvzone.putTextRect(img,f'Waist:{int(waist_Org)}cm',(10,250),scale=2)
                cvzone.putTextRect(img,f'Hand:{int(hand_Org)}cm',(10,340),scale=2)
                cvzone.putTextRect(img,f'Height:{int(shirt_Org)}cm',(10,400),scale=2)
                cvzone.putTextRect(img,f'ankle:{int(ankle_Org)}cm',(10,420),scale=2)
            cv2.imshow("Image",img)
            cv2.waitKey(1)
            i+=1
        cv2.destroyWindow("Image")
    shoulder=(sum(avg_shoulder)/180)*100
    waist=(sum(avg_waist)/180)*100
    hand=(sum(avg_hand)/180)*100
    height=(sum(avg_shirt)/180)*100
    ankle=(sum(avg_ankle)/180)*100
    return render_template('Template/measurement.html',shoulder=shoulder,waist=waist,hand=hand,height=height,ankle=ankle)
@app.route('/tryon',methods=['POST','GET'])
def tryon():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    shirtsfolderpath="Resources"
    listShirts=os.listdir(shirtsfolderpath)

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
        if lmList:
            lm11=lmList[11][0:2]
            lm12=lmList[12][0:2]
            # print(lmList)
            imgShirt=cv2.imread(os.path.join(shirtsfolderpath,listShirts[0]),cv2.IMREAD_UNCHANGED)
            imgShirt=cv2.resize(imgShirt,(0,0),None,0.5,0.5)
            try:
                print(lm12)
                img=cvzone.overlayPNG(img,imgShirt,[lm12[0]-30,lm12[1]-50])
            except:
                pass
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"): 
                break
        cv2.waitKey(1)
    return render_template('Template/login.html')
        
if __name__=='__main__':
    app.run(debug=True,port=5001)