from flask import Flask, request
from image_processing import process_image

app = Flask(__name__)
@app.route('/camera', methods=['POST'])
def process_image_endpoint():
    try:
        snapshot_file = request.files['snapshot']
        if snapshot_file:
            # Save the snapshot to a specific path
            snapshot_path = 'pics/snapshot.jpg'
            snapshot_file.save(snapshot_path)
             # Call the image processing function
            result, status_code = process_image(snapshot_path)
            
            return result, status_code
        else:
            return 'Snapshot file not found', 400
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == "__main__":
    app.run(debug=True)