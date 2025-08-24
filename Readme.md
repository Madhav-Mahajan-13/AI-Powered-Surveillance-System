AI-Powered Surveillance System
This project is a complete implementation of an AI-powered surveillance system using Python. It leverages YOLOv5 for object detection, the SORT algorithm for tracking, and custom logic to detect suspicious activities like abandoned objects and loitering. The entire system is presented through a real-time dashboard built with Streamlit.

Features
Real-Time Object Detection: Uses a pre-trained YOLOv5 model to detect 80 common object classes.

Multi-Object Tracking: Implements the SORT algorithm to assign and maintain a unique ID for each detected object.

Abandoned Object Detection: Flags objects that remain stationary for a specified duration and are not in proximity to any person.

Unusual Movement (Loitering) Detection: Identifies people who remain within a predefined "no-loitering" zone for too long.

Interactive Dashboard: A web-based UI built with Streamlit to display the live video feed, configure parameters, and view real-time alerts.

Project Structure
.
├── main_app.py               # The main Streamlit application file
├── sort_tracker.py           # The SORT tracking algorithm implementation
├── logic_engine.py           # Core logic for abandoned objects and loitering
├── utils.py                  # Helper functions for drawing on frames
├── requirements.txt          # List of Python dependencies
├── videos/                   # Directory for your input videos
│   └── example_video.mp4
└── README.md                 # This file


## Setup Instructions

1.  **Clone the Project:**
    Create a new directory for your project and save all the provided Python files (`main_app.py`, `sort_tracker.py`, `logic_engine.py`, `utils.py`, `requirements.txt`) into it.

2.  **Create a Virtual Environment (Recommended):**
    It's best practice to create a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all the required Python libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Video File:**
    Create a directory named `videos` in your project folder. Place a video file you want to analyze inside it (e.g., `example_video.mp4`). You can find suitable videos from sites like Pexels by searching for "airport," "lobby," or "public square."

## How to Run

Once the setup is complete, you can run the surveillance dashboard with a single command from your terminal:

```bash
streamlit run main_app.py
This will start the Streamlit server and open the application in your web browser. You can then upload a video or use the sample video to see the system in action.