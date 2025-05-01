import cv2
import mediapipe as mp
from mi import face_mesh,hand_model
import eye_ration as er
import head_pose as hp
import hand_pose as hand_pose
import determine as deter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



def frames():
    cap = cv2.VideoCapture(0)
    count=0
    f_no=0
    eye_rate=[]
    pitch=[]
    ya=[]
    dis=[]
    pt=[]
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)
        results_hand =hand_model.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
        
        h, w, _ = image.shape

        if results.multi_face_landmarks:
            index=0
            for face_landmarks in results.multi_face_landmarks:
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                left_ear = er.calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear =er.calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            
                avg_eye_rate=(left_ear+right_ear)/2                    
                projected_pts, (pitch, yaw, roll) =hp.solve_head_pose(face_landmarks, image.shape[1], image.shape[0])
                
                eye_rate.append(avg_eye_rate)
                ya.append(yaw)
                dis.append(0)
                pt.append(pitch)
                
                if results_hand.multi_hand_landmarks:
                    for hand_landmarks in results_hand .multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                        
                        
                        distan=hand_pose.is_hand_near_face(hand_landmarks.landmark,face_landmarks.landmark,w,h)
                        
                    
                        dis[index]=distan
                        index+=1
                        break
            count+=1
          
            deter.then_c(eye_rate,ya,dis,pt,count)
            eye_rate=[]
            dis=[]
            ya=[]
            pt=[]
            if count>=30:
                count=0
            """
            if count>=30:
                deter.then_c(eye_rate,ya,dis,pt)
                eye_rate=[]
                dis=[]
                ya=[]
                pt=[]
                count=0
            """
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        f_no+=1


      
        if cv2.waitKey(1)  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
frames()
