# Traffic Management System with Reinforcement Learning and Computer Vision

## Overview
This project integrates Deep Q-Network (DQN) reinforcement learning with the SUMO (Simulation of Urban MObility) traffic simulator for dynamic traffic signal control at intersections. Additionally, it incorporates YOLOv8 computer vision models for real-time traffic monitoring and detection of congestion and road hazards.

## Features
- **Reinforcement Learning for Traffic Signal Control**: Utilizes DQN algorithm with SUMO to optimize traffic signal timings based on real-time conditions, reducing congestion and improving traffic flow.
  
- **YOLOv8 for Traffic Monitoring**: Integrates YOLOv8 model for efficient detection of vehicles and objects within specified Regions of Interest (ROIs), providing valuable insights into traffic density and movement patterns.
  
- **Custom YOLOv8 Models**:
  - *Congestion Detection*: Custom-trained YOLOv8 model identifies areas of traffic congestion, enabling proactive traffic management measures.
  - *Pothole Detection*: Detects road defects like potholes, facilitating timely maintenance efforts to ensure road safety.
 
## Results
- ![image](https://github.com/pranavsrinivasa/Traffic-Optimization-Projects/assets/126983069/aa624a3f-2e82-4f43-a14e-8cd58f6a8a81)
- ![image](https://github.com/pranavsrinivasa/Traffic-Optimization-Projects/assets/126983069/30ab05a4-3f08-4f7b-9e32-278538460b41)
  ![image](https://github.com/pranavsrinivasa/Traffic-Optimization-Projects/assets/126983069/d5225631-d1fa-4fba-b9ce-233f96e196e2)
- ![image](https://github.com/pranavsrinivasa/Traffic-Optimization-Projects/assets/126983069/ac48f92b-6566-4657-ab1a-da21c79c0d4c)
- ![image](https://github.com/pranavsrinivasa/Traffic-Optimization-Projects/assets/126983069/6ec29e1f-5aa2-4a35-b6cb-e77944488c04)


## Usage
1. **Installation**:
   - Install required dependencies: SUMO, PyTorch, OpenCV, etc.
   - Download pre-trained YOLOv8 weights or train custom models as needed.
   - ```pip install requirements.txt```
   - Install Ollama from https://ollama.com/ and hence install mistral model on cmd with the command
     ```
     ollama pull mistral
     ```
     
2. **Configuration**:
   - Configure SUMO simulation environment and traffic scenarios.
   - Set up YOLOv8 models for traffic monitoring and detection tasks.
   - Run Ollama application
   
3. **Execution**:
   - Run the integrated system to initiate traffic simulation and monitoring.
   - Analyze results for traffic signal optimization, congestion detection, and pothole identification
   - To run the app :
     ``` streamlit run app.py ```
