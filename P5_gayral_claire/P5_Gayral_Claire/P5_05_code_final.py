import pickle
from bs4 import BeautifulSoup
import pandas as pd
# import sys 
# post_inputs = sys.argv[1]

## load vectorizer :
with open('preprocess_txt.pkl', 'rb') as f:
    countvec = pickle.load(f)
## load classifier :
with open('best_clf.pkl', 'rb') as f:
    clf = pickle.load(f)

## filter language 
map_lang_dict = {"Qt_framework": ["Qt_framework","qt", "qt4", "qt5", "qt-creator", "pyqt"], 
 "VisualBasic": ["VisualBasic", "VB", "vb", "Vb", "VB.net"],
 "VisualStudio": ["VisualStudio","vs"], 
 "clang": ["clang","c", "C"],
 "cplusplus": ["cplusplus", "c++", "C++", "c++14", "C++14", "c++17", "C++17", "c++98", 
               "C++98", "c++11", "C++11", "c++20", "clang++", "libstdc++",
               "libs++", "c++-cli", "c++-faq", "c++-tr2", "c++builder"], 
 "csharp": ["csharp", "c#", "C#", "c#-2.0", "c#-3.0", "c#-4.0"], 
 "d3js": ["d3js","d3.js", "d3\\.js", "nvd3.js"], 
 "dotnet": ["dotnet",".net", ".net-1.0", ".net-1.1", ".net-2.0", ".net-3.0", ".net-3.5", 
            ".net-assembly", ".net-attributes", ".net-client-profile", 
            ".net-framework-version"],
 "goLang": ["goLang","Go", "go"],
 "gplusplus": ["gplusplus","g++5.1", "g++4.8", "g++", "gcc4.8", "gcc", "gcc10", "g++10"],
 "input_output": ["input_output","io"], 
 "javascript": ["javascript","js"], 
 "rLang": ["rLang","r", "R"], 
 "sql": ["ansi-sql-92", "bpgsql", "dynamic-sql", "entity-sql", "linq-to-sql", "mysql", 
         "mysql-error-1005", "mysql-error-1062", "mysql-error-1093", "mysql-error-1111", 
         "mysql-error-1451", "mysql-error-2006", "mysql-management", "mysql-parameter", 
         "mysql-workbench", "mysqldump", "mysqli", "nosql", "oracle-sqldeveloper", "plsql",
         "postgresql", "psql", "sql", "sql-convert", "sql-delete", "sql-execution-plan", 
         "sql-injection", "sql-insert", "sql-like", "sql-loader", "sql-order-by", "sql-server",
         "sql-server-2000", "sql-server-2005", "sql-server-2005-express", "sql-server-2008",
         "sql-server-agent", "sql-server-ce", "sql-server-express", "sql-server-mobile", 
         "sql-types", "sql-update", "sqlalchemy", "sqlclr", "sqlcommand", "sqlcompare", 
         "sqlconnection", "sqldatatypes", "sqldmo", "sqlexception", "sqlite", "sqlncli", "sqlplus", "tsql"], 
 "vuejs": [ "vuejs","vue.js", "vuejs2", "vue-component", "vuex", "vuetify.js", "vue-router"]
}
lang_dict = {}
for k,v in map_lang_dict.items() :
    for lang in v :
        lang_dict[lang] = k
        
## preprocess : 
def extract_text_btw_html(html):
    ## automatic clean of text : remove html balises
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    return(list(soup.stripped_strings))

def replace_with_dict_val(my_list, my_dict) :
    ## replace values in list if they are in dict keys
    for i in range(len(my_list)) :
        t = my_list[i]
        if t in lang_dict.keys():
            t = lang_dict[t]
    return(my_list)

def main(post_input):
    '''
    Main function of code final, predict the best tag for post
    
    Parameters:
    -----------------------------------------
    post_input (str) : the text of the post to be tagged
    
    Returns:
    -----------------------------------------
    list of most probable tags (str) associate to the post
    '''
    ## init preprocess
    x_test = []
    text = post_input
    ## preprocess - remove non textual data
    list_text = extract_text_btw_html(text)
    text =  ' '.join(list_text)
    tokens = text.split(" ")
    ## preprocess - filter langage versions
    tokens = replace_with_dict_val(tokens, lang_dict)
    x_test.append(" ".join(tokens))
    ## right shape preprocessed data : 
    x_test = pd.Series(x_test)
    ## count-vectorize :
    X_test = countvec.transform(x_test).toarray()
    ## classif prediction : 
    return(clf.predict(X_test))
 