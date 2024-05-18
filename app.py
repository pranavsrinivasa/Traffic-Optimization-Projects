from ultralytics import YOLO
import base64
import cv2
import io
import numpy as np
from ultralytics.utils.plotting import Annotator
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def set_background(image_file1,image_file2):
    
    with open(image_file1, "rb") as f:
        img_data1 = f.read()
    b64_encoded1 = base64.b64encode(img_data1).decode()
    with open(image_file2, "rb") as f:
        img_data2 = f.read()
    b64_encoded2 = base64.b64encode(img_data2).decode()
    style = f"""
        <style>
        .stApp{{
            background-image: url(data:image/png;base64,{b64_encoded1});
            background-size: cover;
            
        }}
        .st-emotion-cache-6qob1r{{
            background-image: url(data:image/png;base64,{b64_encoded2});
            background-size: cover;
            border: 5px solid rgb(14, 17, 23);
            
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('pngtree-city-map-navigation-interface-picture-image_1833642.png','2024-05-18_14-57-09_5235.png')

st.title("Traffic Flow and Optimization Toolkit")

sb = st.sidebar # defining the sidebar

sb.markdown("🛰️ **Navigation**")
page_names = ["PS1", "PS2", "PS3","Chat with Results"]
page = sb.radio("", page_names, index=0)
st.session_state['n'] = sb.slider("Number of ROIs",1,5)

if page == 'PS1':
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read()) 
        temporary_location = "temp_PS1.mp4"

        with open(temporary_location, 'wb') as out:  
            out.write(g.read())  
        out.close()
        model = YOLO('PS1\yolov8n.pt')
        if 'roi_list1' not in st.session_state:
            st.session_state['roi_list1'] = []
        if "all_rois1" not in st.session_state:    
            st.session_state['all_rois1'] = []
        classes = model.names

        done_1 = st.button('Selection Done')

        while len(st.session_state["all_rois1"]) < st.session_state['n']:
            cap = cv2.VideoCapture('temp_PS1.mp4')
            while not done_1:
                ret,frame=cap.read()
                cv2.putText(frame,'SELECT ROI',(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4)
                if not ret:
                    st.write('ROI selection has concluded')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                value = streamlit_image_coordinates(frame,key='numpy',width=750)
                st.session_state["roi_list1"].append([int(value['x']*2.55),int(value['y']*2.55)])
                st.write(st.session_state["roi_list1"])
                if cv2.waitKey(0)&0xFF==27:
                    break
            cap.release()
            st.session_state["all_rois1"].append(st.session_state["roi_list1"])
            st.session_state["roi_list1"] = []
            done_1 = False

        st.write('ROI indices: ',st.session_state["all_rois1"][0])



        cap = cv2.VideoCapture('temp_PS1.MP4')
        st.write("Detection started")
        st.session_state['fps'] = cap.get(cv2.CAP_PROP_FPS)
        st.write(f"FPS OF VIDEO: {st.session_state['fps']}")
        avg_list = []
        count = 0
        frame_placeholder = st.empty()
        st.session_state["data1"] = {}
        for i in range(len(st.session_state["all_rois1"])):
            st.session_state["data1"][f"ROI{i}"] = []
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            count += 1
            if count % 3 != 0:
                continue
            k = 0
            for roi_list_here1 in st.session_state["all_rois1"]:
                max = [0,0]
                min = [10000,10000]
                roi_list_here = roi_list_here1[1:]
                for i in range(len(roi_list_here)):
                    if roi_list_here[i][0] > max[0]:
                        max[0] = roi_list_here[i][0]
                    if roi_list_here[i][1] > max[1]:
                        max[1] = roi_list_here[i][1]
                    if roi_list_here[i][0] < min[0]:
                        min[0] = roi_list_here[i][0]
                    if roi_list_here[i][1] < min[1]:
                        min[1] = roi_list_here[i][1]
                frame_cropped = frame[min[1]:max[1],min[0]:max[0]]
                roi_corners = np.array([roi_list_here],dtype=np.int32)
                mask = np.zeros(frame.shape,dtype=np.uint8)
                mask.fill(255)
                channel_count = frame.shape[2]
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask,roi_corners,0)
                mask_cropped = mask[min[1]:max[1],min[0]:max[0]]
                roi = cv2.bitwise_or(frame_cropped,mask_cropped)

                #roi = frame[roi_list_here[0][1]:roi_list_here[1][1],roi_list_here[0][0]:roi_list_here[1][0]]
                number = []
                results = model.predict(roi)
                for r in results:
                    boxes = r.boxes
                    counter = 0
                    for box in boxes:
                        counter += 1
                        name = classes[box.cls.numpy()[0]]
                        conf = str(round(box.conf.numpy()[0],2))
                        text = name+""+conf
                        bbox = box.xyxy[0].numpy()
                        cv2.rectangle(frame,(int(bbox[0])+min[0],int(bbox[1])+min[1]),(int(bbox[2])+min[0],int(bbox[3])+min[1]),(0,255,0),2)
                        cv2.putText(frame,text,(int(bbox[0])+min[0],int(bbox[1])+min[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
                    number.append(counter)
                avg = sum(number)/len(number)
                stats = str(round(avg,2))
                if count%10 == 0:
                    st.session_state["data1"][f"ROI{k}"].append(avg)
                    k+=1
                cv2.putText(frame,stats,(min[0],min[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),4)
                cv2.polylines(frame,roi_corners,True,(255,0,0),2)
            cv2.putText(frame,'The average number of vehicles in the Regions of Interest',(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4)
            frame_placeholder.image(frame,channels='BGR')
        cap.release()
        df = pd.DataFrame(st.session_state["data1"])
        df.to_csv('PS1.csv', sep='\t', encoding='utf-8')
    else:
        st.error('PLEASE UPLOAD AN IMAGE OF THE FORMAT JPG,JPEG OR PNG', icon="🚨")

elif page == "PS2":
    uploaded_file1 = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file1 is not None:
        g = io.BytesIO(uploaded_file1.read()) 
        temporary_location = "temp_PS2.mp4"

        with open(temporary_location, 'wb') as out:  
            out.write(g.read())  
        out.close()
        model1 = YOLO("PS1\yolov8n.pt")
        model2 = YOLO(r"PS3\best.pt")
        if 'roi_list2' not in st.session_state:
            st.session_state['roi_list2'] = []
        if "all_rois2" not in st.session_state:    
            st.session_state['all_rois2'] = []
        classes = model1.names

        done_2 = st.button('Selection Done')

        while len(st.session_state["all_rois2"]) < st.session_state['n']:
            cap = cv2.VideoCapture('temp_PS2.mp4')
            while not done_2:
                ret,frame=cap.read()
                cv2.putText(frame,'SELECT ROI',(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4)
                if not ret:
                    st.write('ROI selection has concluded')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                value = streamlit_image_coordinates(frame,key='numpy',width=750)
                st.session_state["roi_list2"].append([int(value['x']*2.5),int(value['y']*2.5)])
                st.write(st.session_state["roi_list2"])
                if cv2.waitKey(0)&0xFF==27:
                    break
            cap.release()
            st.session_state["all_rois2"].append(st.session_state["roi_list2"])
            st.session_state["roi_list2"] = []
            done_2 = False

        st.write('ROI indices: ',st.session_state["all_rois2"][0])



        cap = cv2.VideoCapture('temp_PS2.MP4')
        st.write("Detection started")
        avg_list = []
        count = 0
        frame_placeholder = st.empty()
        st.session_state.data = {}
        for i in range(len(st.session_state["all_rois2"])):
            st.session_state["data"][f"ROI{i}"] = []
        for i in range(len(st.session_state['all_rois2'])):
            st.session_state.data[f"ROI{i}"] = []
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:
                break
            count += 1
            if count % 3 != 0:
                continue
            # rois = []
            k = 0
            for roi_list_here1 in st.session_state["all_rois2"]:
                max = [0,0]
                min = [10000,10000]
                roi_list_here = roi_list_here1[1:]
                for i in range(len(roi_list_here)-1):
                    if roi_list_here[i][0] > max[0]:
                        max[0] = roi_list_here[i][0]
                    if roi_list_here[i][1] > max[1]:
                        max[1] = roi_list_here[i][1]
                    if roi_list_here[i][0] < min[0]:
                        min[0] = roi_list_here[i][0]
                    if roi_list_here[i][1] < min[1]:
                        min[1] = roi_list_here[i][1]
                frame_cropped = frame[min[1]:max[1],min[0]:max[0]]
                roi_corners = np.array([roi_list_here],dtype=np.int32)
                mask = np.zeros(frame.shape,dtype=np.uint8)
                mask.fill(255)
                channel_count = frame.shape[2]
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask,roi_corners,0)
                mask_cropped = mask[min[1]:max[1],min[0]:max[0]]
                roi = cv2.bitwise_or(frame_cropped,mask_cropped)

                #roi = frame[roi_list_here[0][1]:roi_list_here[1][1],roi_list_here[0][0]:roi_list_here[1][0]]
                number = []
                results = model1.predict(roi)
                results_pothole = model2.predict(source=frame)
                for r in results:
                    boxes = r.boxes
                    counter = 0
                    for box in boxes:
                        counter += 1
                        name = classes[box.cls.numpy()[0]]
                        conf = str(round(box.conf.numpy()[0],2))
                        text = name+conf
                        bbox = box.xyxy[0].numpy()
                        cv2.rectangle(frame,(int(bbox[0])+min[0],int(bbox[1])+min[1]),(int(bbox[2])+min[0],int(bbox[3])+min[1]),(0,255,0),2)
                        cv2.putText(frame,text,(int(bbox[0])+min[0],int(bbox[1])+min[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),2)
                    number.append(counter)
                for r in results_pothole:
                    masks = r.masks
                    boxes = r.boxes.cpu().numpy()
                    xyxys = boxes.xyxy
                    confs = boxes.conf
                    if masks is not None:
                        shapes = np.ones_like(frame)
                        for mask,conf,xyxy in zip(masks,confs,xyxys):
                            polygon = mask.xy[0]
                            if conf >= 0.49 and len(polygon)>=3:
                                cv2.fillPoly(shapes,pts=np.int32([polygon]),color=(0,0,255,0.5))
                                frame = cv2.addWeighted(frame,0.7,shapes,0.3,gamma=0)
                                cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,0,255),2)
                                cv2.putText(frame,'Pothole '+str(conf),(int(xyxy[0]),int(xyxy[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

                avg = sum(number)/len(number)
                stats = str(round(avg,2))
                if count % 10 == 0:
                    st.session_state.data[f"ROI{k}"].append(avg)
                    k+=1
                cv2.putText(frame,stats,(min[0],min[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),4)
                cv2.polylines(frame,roi_corners,True,(255,0,0),2)
                if counter >= 5:
                    cv2.putText(frame,'!!CONGESTION MORE THAN '+str(counter)+' Objects',(min[0]+20,min[1]+20),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),4)
                cv2.polylines(frame,roi_corners,True,(255,0,0),2)
                cv2.putText(frame,'Objects in the Regions of Interest',(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4)
            frame_placeholder.image(frame,channels='BGR')
        cap.release()

        df = pd.DataFrame(st.session_state.data)
        df.to_csv('PS2.csv', sep='\t', encoding='utf-8')

    else:
        st.error('PLEASE UPLOAD AN IMAGE OF THE FORMAT JPG,JPEG OR PNG', icon="🚨")

elif page == "PS3":
    st.header("hello world")

elif page == "Chat with Results":
    st.title('Chat with the Results')
    st.write("Please upload the relevant CSV data to get started")
    reload = st.button('Reload')
    if 'isran' not in st.session_state or reload == True:
        st.session_state['isran'] = False


    uploaded_file = st.file_uploader('Choose your .csv file', type=["csv"])
    if uploaded_file is not None and st.session_state['isran'] == False:
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = CSVLoader('temp.csv')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OllamaEmbeddings(model='mistral')
        st.session_state.vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
        st.session_state['isran'] = True

    if st.session_state['isran'] == True:
        st.write("Embedding created")

    def fdocs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def llm(question,context):
        formatted_prompt = f"Question: {question}\n\nContext:{context}"
        response = ollama.chat(model='mistral', messages=[
            {
                'role': 'user',
                'content': formatted_prompt
            },
            ])
        return response['message']['content']



    def rag_chain(question): 
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(question)
        formatted_context = fdocs(retrieved_docs)
        return llm(question,formatted_context)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Say something")
    response = rag_chain(prompt)
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        st.session_state.messages.append({'role':'AI','content':response})
        st.chat_message('AI').markdown(response)
