from flask_sqlalchemy import SQLAlchemy
from scipy.datasets import face

db = SQLAlchemy()

def init_db(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///home_gallery.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()

class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey("photo.id"))
    person_id = db.Column(db.Integer, db.ForeignKey("person.id"))
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    w = db.Column(db.Integer)
    h = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    face_rel_x = db.Column(db.Float)
    face_rel_y = db.Column(db.Float)
    face_rel_w = db.Column(db.Float)
    face_rel_h = db.Column(db.Float)

    photo = db.relationship("Photo", backref="faces")
    person = db.relationship("Person", backref="faces")