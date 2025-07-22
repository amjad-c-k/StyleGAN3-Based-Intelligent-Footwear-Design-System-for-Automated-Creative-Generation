import pyrebase

#Configure and Connext to Firebase

firebaseConfig = {
  'apiKey': "AIzaSyCxCo10M1wSmHuqkTKDmPt-0wDgHX1IDtw",
  'authDomain': "generativeaishoedesigner-94f9f.firebaseapp.com",
  'projectId': "generativeaishoedesigner-94f9f",
  'storageBucket': "generativeaishoedesigner-94f9f.firebasestorage.app",
  'messagingSenderId': "601866275210",
  'appId': "1:601866275210:web:ce1f35f7423c58f6a4e073",
  'measurementId': "G-HCQ0EQF9QC"
}


firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()

#Login function

def login():
    print("Log in...")
    email=input("Enter email: ")
    password=input("Enter password: ")
    try:
        login = auth.sign_in_with_email_and_password(email, password)
        print("Successfully logged in!")
        # print(auth.get_account_info(login['idToken']))
       # email = auth.get_account_info(login['idToken'])['users'][0]['email']
       # print(email)
    except:
        print("Invalid email or password")
    return

#Signup Function

def signup():
    print("Sign up...")
    email = input("Enter email: ")
    password=input("Enter password: ")
    try:
        user = auth.create_user_with_email_and_password(email, password)
        ask=input("Do you want to login?[y/n]")
        if ask=='y':
            login()
    except: 
        print("Email already exists")
    return

#Main

ans=input("Are you a new user?[y/n]")

if ans == 'n':
    login()
elif ans == 'y':
    signup()
