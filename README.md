# ARISA Web application to manage home photos with face recognition

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask application:
   ```
   python API/api_app.py
   ```

2. The API will be available at `http://127.0.0.1:5000`.

## API Endpoints
### 1. Face Detection
- **Endpoint:** `/api/detect_faces`
- **Method:** POST
- **Description:** Accepts an image upload and returns the locations of detected faces in JSON format.

### 2. Face Training
- **Endpoint:** `/api/train`
- **Method:** POST
- **Description:** Accepts an image of a face along with the person's name to train the face recognition model.

### 3. Face Recognition
- **Endpoint:** `/api/predict`
- **Method:** POST
- **Description:** Accepts an image of a face and returns the predicted name of the person in the image and confidence.