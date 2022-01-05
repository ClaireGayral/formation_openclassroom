from flask import Flask
from flask import render_template
## request html :
from flask import request
from flask import url_for
from flask import flash
from flask import redirect

app = Flask(__name__)
app.config['SECRET_KEY'] = '1364b5295c0fba438a24c5d79ed621e13bd1fd1bc008fc9d'


messages = [{'title': 'Présentation',
             'content': 'Cette API permet de prédire les tags les plus probables associés à un texte.'},
            {'title': 'Utilisation',
             'content': "Pour l'utiliser, il suffit de coller le contenu de la publication à tagguer dans le lien suivant. Elle peut être avec balise (format html par exemple) ou juste les mots."}
            ]

@app.route('/')
def index():
    return render_template('index.html', messages=messages)
    
@app.route('/create/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('Merci de remplir le champ avant de lancer le programme !')
        else:
            messages.append({'content': content})
            return redirect(url_for('index'))

    return render_template('create.html')
    
    
    
    
    
    
