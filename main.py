import csv
from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
import json
from datetime import datetime
import math
import plotly.graph_objs as go
import plotly.offline as pyo
import face_recognition
import cv2
import numpy as np
import plotly.express as px
import pandas as pd

local_server = True
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

db = SQLAlchemy()
app = Flask(__name__)

if local_server:
    app.config["SQLALCHEMY_DATABASE_URI"] = params['local_uri']
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = params['prod_uri']

db.init_app(app)


class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String, nullable=False)
    image = db.Column(db.Text, nullable=False)
    type = db.Column(db.String, nullable=False)


class Entry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    faceID = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String, nullable=False)
    inTime = db.Column(db.DateTime, nullable=False, default=datetime.strftime(datetime.today(), "%b %d %Y"))
    outTime = db.Column(db.DateTime, nullable=False, default=datetime.strftime(datetime.today(), "%b %d %Y"))
    date = db.Column(db.Date, nullable=False)
    type = db.Column(db.String, nullable=False)


@app.route("/")
def hello():
    student = Face.query.filter_by(type="student").all()
    staff = Face.query.filter_by(type="staff").all()
    visitor = Face.query.filter_by(type="visitor").all()
    unknown = Face.query.filter_by(type="unknown").all()

    student = len(student)
    staff = len(staff)
    visitor = len(visitor)
    unknown = len(unknown)

    faces = Entry.query.filter_by().all()
    last = math.ceil(len(faces) / int(params['no_of_rows']))
    # [0:params['no_of_rows']]
    page = request.args.get('page')
    if not str(page).isnumeric():
        page = 1
    page = int(page)
    faces = faces[
            (page - 1) * int(params['no_of_rows']): (page - 1) * int(params['no_of_rows']) + int(params['no_of_rows'])]
    if page == 1:
        prev = "#"
        nex = "/?page=" + str(page + 1)
    elif page == last:
        prev = "/?page=" + str(page - 1)
        nex = "#"
    else:
        prev = "/?page=" + str(page - 1)
        nex = "/?page=" + str(page + 1)

    # generate plotly graph
    data = [
        go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines', name='line'),
        go.Bar(x=[1, 2, 3, 4], y=[14, 15, 16, 17], name='bar')
    ]
    layout = go.Layout(title='My Plot')
    fig = go.Figure(data=data, layout=layout)
    plot_div = pyo.plot(fig, output_type='div')

    fig2 = go.Figure(data=data, layout=layout)
    plot2_div = pyo.plot(fig2, output_type='div')

    result_set = db.session.query(Entry).all()
    # long_df = px.Entry.query.filter_by().all()
    df = pd.DataFrame([(row.id, row.faceID, row.name, row.age, row.gender, row.inTime, row.outTime, row.date, row.type) for row in result_set], columns=['id', 'faceID', 'name', 'age', 'gender', 'inTime', 'outTime', 'date', 'type'])

    fig3 = px.bar(df, x="date", y="type", color="type", title="Long-Form Input", width=1000, template="plotly_dark", color_discrete_map={'student': 'teal', 'staff': 'turquoise','visitor': 'aqua'})
    # fig3.update_traces(width=100)
    fig3.update_layout(title_x = 0.5, paper_bgcolor="rgba(55,57,61,1)", plot_bgcolor="rgba(0,0,0,0)")
    # fig.show()
    plot3_div = pyo.plot(fig3, output_type='div')

    return render_template('index.html', params=params, faces=faces, prev=prev, nex=nex, student=student, staff=staff,
                           visitor=visitor, unknown=unknown, plot3_div=plot3_div)




@app.route("/members")
def members():
    people = Face.query.filter_by().all()
    # [0:params['no_of_rows']]

    last = math.ceil(len(people) / int(params['no_of_rows']))
    # [0:params['no_of_rows']]
    sheet = request.args.get('sheet')
    if not str(sheet).isnumeric():
        sheet = 1
    sheet = int(sheet)
    people = people[
             (sheet - 1) * int(params['no_of_rows']): (sheet - 1) * int(params['no_of_rows']) + int(
                 params['no_of_rows'])]
    if sheet == 1:
        before = "#"
        after = "/members?sheet=" + str(sheet + 1)
    elif sheet == last:
        before = "/members?sheet=" + str(sheet - 1)
        after = "#"
    else:
        before = "/members?sheet=" + str(sheet - 1)
        after = "/members?sheet=" + str(sheet + 1)

    return render_template('members.html', params=params, people=people, before=before, after=after)


@app.route("/edit/<string:id>", methods=['GET', 'POST'])
def edit_data(id):
    if request.method == 'POST':
        pic = request.files["image"]
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        type = request.form["type"]

        face = Face.query.filter_by(id=id).first()
        face.name = name
        face.age = age
        face.gender = gender
        face.image = pic.read()
        face.type = type
        db.session.commit()
        return redirect('/edit/' + id)
    face = Face.query.filter_by(id=id).first()
    return render_template('edit.html', params=params, face=face)


@app.route("/delete/<string:id>", methods=['GET', 'POST'])
def delete(id):
    face = Face.query.filter_by(id=id).first()
    db.session.delete(face)
    db.session.commit()
    people = Face.query.filter_by().all()[0:params['no_of_rows']]
    return render_template('members.html', params=params, people=people)


@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        pic = request.files["image"]

        face = Face(
            name=request.form["name"],
            age=request.form["age"],
            gender=request.form["gender"],
            image=pic.read(),
            type=request.form["type"],
        )
        db.session.add(face)
        db.session.commit()
    return render_template('form.html')


@app.route("/setting")
def setting():
    return render_template('setting.html', params=params)


@app.route("/face")
def recognize_faces():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    known_face_encodings = []
    known_face_names = []
    known_id = []
    known_age = []
    known_gender = []
    known_type = []

    with app.app_context():
        face = Face.query.all()

    for person in face:
        image_np = np.frombuffer(person.image, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(person.name)
        known_id.append(person.id)
        known_age.append(person.age)
        known_gender.append(person.gender)
        known_type.append(person.type)

    students = known_face_names.copy()
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    #
    # f = open(current_date+'.csv','w+',newline= '')
    # lnwriter = csv.writer(f)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    id = known_id[best_match_index]
                    age = known_age[best_match_index]
                    gender = known_gender[best_match_index]
                    type = known_type[best_match_index]

                if name in known_face_names:
                    if name in students:
                        students.remove(name)
                        with app.app_context():
                            new_face = Entry(faceID=id, name=name, age=age, gender=gender, inTime=now, outTime=now,
                                             date=current_date, type=type)
                            db.session.add(new_face)
                            db.session.commit()

                face_names.append(name)


        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    # f.close()
    return render_template("face.html", facen=face_names)



@app.route("/help")
def helps():
    return render_template('help.html', params=params)


app.run(debug=True)
