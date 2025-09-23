import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet
from ARISA_DSML.resolve import detect_faces, convert_results
from ARISA_DSML.helpers import FaceRecognizer
from ARISA_DSML.config import init_db, db, Photo, Person, Face
from ARISA_DSML.preproc import preprocess_image
from PIL import Image
import numpy as np

embedder = FaceNet()
rec = FaceRecognizer(embedder)
rec.load()
print("Data loaded")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "API/static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

init_db(app)


@app.route("/")
def gallery():
    photos = Photo.query.all()
    return render_template("gallery.html", photos=photos)

@app.route("/photo/<int:photo_id>")
def photo_detail(photo_id):
    photo = Photo.query.get(photo_id)
    return render_template("photo.html", photo=photo)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        image_file = request.files["file"]
        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(filepath)

            image = preprocess_image(image_file)
            face_locations = detect_faces(image)
            image_height, image_width = image.shape[:2]
            photo = Photo(filename=filename)
            db.session.add(photo)
            db.session.commit()

            for face in face_locations:
                face["box"] = (face["box"][0], face["box"][1], face["box"][2], face["box"][3])
                face_image = image[face["box"][1]:face["box"][1]+face["box"][3], face["box"][0]:face["box"][0]+face["box"][2]]
                face_image = Image.fromarray(face_image)
                face_image = face_image.resize((160, 160))
                face_image = np.array(face_image)
                label, confidence = rec.predict([face_image])
                print(label, confidence)
                face["name"] = label[0] if label else "Unknown"
                face["confidence"] = confidence[0] if confidence else 0.0
                person = Person.query.filter_by(name=face["name"]).first()
                if not person:
                    person = Person(name=face["name"])
                    db.session.add(person)
                    db.session.commit()

                db.session.add(Face(
                    photo_id=photo.id,
                    person_id=person.id,
                    x=face["box"][0], y=face["box"][1],
                    w=face["box"][2], h=face["box"][3],
                    confidence=face["confidence"],
                    face_rel_x=face["box"][0] / image_width,
                    face_rel_y=face["box"][1] / image_height,
                    face_rel_w=face["box"][2] / image_width,
                    face_rel_h=face["box"][3] / image_height
                ))
            db.session.commit()

            return redirect(url_for("gallery"))

    return render_template("upload.html")


@app.route("/api/learn_person/<int:photo_id>/<int:face_id>", methods=["POST"])
def learn_person(photo_id, face_id):
    photo = Photo.query.get_or_404(photo_id)
    face = Face.query.get_or_404(face_id)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], photo.filename)
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    x, y, w, h = face.x, face.y, face.w, face.h
    face_image = image[y:y+h, x:x+w]
    face_image = Image.fromarray(face_image)
    face_image = face_image.resize((160, 160))
    face_image = np.array(face_image)
    name = request.json.get("name")
    if not name:
        return jsonify({"error": "Name is required"}), 400

    print(f"New face {name} to learn. Current '{face.person.name if face.person else 'Unknown'}'")
    success = rec.add_person([face_image], [name])
    
    if success:
        rec.save()
        person = Person.query.filter_by(name=name).first()
        if not person:
            person = Person(name=name)
            db.session.add(person)
            db.session.commit()
        else:
            person.name = name
            db.session.commit()
        face.person_id = person.id
        db.session.commit()
        return jsonify({"status": "ok", "id": person.id, "name": person.name}), 200
    else:
        return jsonify({"error": "Failed to learn person"}), 500


if __name__ == '__main__':
    app.run(debug=True)
