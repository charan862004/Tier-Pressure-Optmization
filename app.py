import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import base64

# Function to convert local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your image
img_path = r"C:\Users\DELL\Downloads\generated-image.png"
img_base64 = get_base64_of_bin_file(img_path)

# CSS for background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# ----------------------------
# Objective and derivatives
# ----------------------------
def f(P, A=1.0, D=0.5, k=0.1, C=0.01, Ps=35):
    return A + D*np.exp(-k*P) + C*(P-Ps)**2

def f1(P, A=1.0, D=0.5, k=0.1, C=0.01, Ps=35):
    return -k*D*np.exp(-k*P) + 2*C*(P-Ps)

def f2(P, A=1.0, D=0.5, k=0.1, C=0.01, Ps=35):
    return k**2*D*np.exp(-k*P) + 2*C

# ----------------------------
# Algorithms
# ----------------------------
def exhaustive_search(a,b,n=201):
    xs = np.linspace(a,b,n)
    vals = f(xs)
    i = int(np.argmin(vals))
    return xs[i], vals[i]

def bounding_phase(a,b,x0,delta):
    xk=x0; k=0
    while True:
        xkp1=xk+(2**k)*delta
        if xkp1>b or f(xkp1)>=f(xk):
            return (xk+xkp1)/2, f((xk+xkp1)/2)
        xk=xkp1; k+=1

def interval_halving(a,b,eps=1e-3):
    while (b-a)>eps:
        L=b-a; x1=a+L/4; xm=(a+b)/2; x2=b-L/4
        f1v,fm,f2v=f(x1),f(xm),f(x2)
        if f1v<fm: b=xm
        elif f2v<fm: a=xm
        else: a,b=x1,x2
    x=(a+b)/2; return x,f(x)

def fibonacci_search(a,b,n=20):
    F=[1,1]
    for _ in range(2,n+2): F.append(F[-1]+F[-2])
    x1=a+(F[n-1]/F[n+1])*(b-a); x2=a+(F[n]/F[n+1])*(b-a)
    f1v,f2v=f(x1),f(x2)
    for k in range(1,n):
        if f1v<f2v:
            b=x2; x2=x1; f2v=f1v
            x1=a+(F[n-k-1]/F[n-k+1])*(b-a); f1v=f(x1)
        else:
            a=x1; x1=x2; f1v=f2v
            x2=a+(F[n-k]/F[n-k+1])*(b-a); f2v=f(x2)
    x=(a+b)/2; return x,f(x)

def golden_section(a,b,eps=1e-3):
    phi=(1+np.sqrt(5))/2; resphi=2-phi
    x1=a+resphi*(b-a); x2=b-resphi*(b-a)
    f1v,f2v=f(x1),f(x2)
    while abs(b-a)>eps:
        if f1v<f2v: b,x2,f2v=x2,x1,f1v; x1=a+resphi*(b-a); f1v=f(x1)
        else: a,x1,f1v=x1,x2,f2v; x2=b-resphi*(b-a); f2v=f(x2)
    x=(a+b)/2; return x,f(x)

def newton_raphson(a,b,x0,eps=1e-6):
    x=x0
    for _ in range(100):
        d1,d2=f1(x),f2(x)
        if abs(d1)<eps: break
        x1=x-d1/d2
        if x1<a: x1=a
        if x1>b: x1=b
        if abs(x1-x)<eps: x=x1; break
        x=x1
    return x,f(x)

def bisection_on_derivative(a,b,eps=1e-6):
    for _ in range(1000):
        m=(a+b)/2; g=f1(m)
        if abs(g)<eps: return m,f(m)
        if g<0: a=m
        else: b=m
    return (a+b)/2,f((a+b)/2)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Tire Pressure Optimization")

a = st.number_input("Minimum Pressure",0)
b = st.number_input("Maximum Pressure",0)

alg = st.selectbox("Choose Algorithm",[
    "Exhaustive Search","Bounding Phase","Interval Halving",
    "Fibonacci Search","Golden Section",
    "Newton–Raphson (double derivative)","Bisection on f'(x)"
])

# Extra params (show only when needed)
n=None; delta=None; x0=None; eps=None
if alg=="Exhaustive Search":
    n=st.number_input("Number of grid points",0)
elif alg=="Bounding Phase":
    x0=st.number_input("Initial Guess",0)
    delta=st.number_input("Delta",0)
elif alg=="Interval Halving":
    eps=st.number_input("Tolerance",0.01)
elif alg=="Fibonacci Search":
    n=st.number_input("Number n",0)
elif alg=="Golden Section":
    eps=st.number_input("Tolerance",0.01)
elif alg=="Newton–Raphson (double derivative)":
    x0=st.number_input("Initial Guess",0)
    eps=st.number_input("Tolerance",0.0001)
elif alg=="Bisection on f'(x)":
    eps=st.number_input("Tolerance",0.0001)

if st.button("Run"):
    if alg=="Exhaustive Search": x,y=exhaustive_search(a,b,n)
    elif alg=="Bounding Phase": x,y=bounding_phase(a,b,x0,delta)
    elif alg=="Interval Halving": x,y=interval_halving(a,b,eps)
    elif alg=="Fibonacci Search": x,y=fibonacci_search(a,b,n)
    elif alg=="Golden Section": x,y=golden_section(a,b,eps)
    elif alg=="Newton–Raphson (double derivative)": x,y=newton_raphson(a,b,x0,eps)
    else: x,y=bisection_on_derivative(a,b,eps)

    st.write(f"**Optimal Pressure:** {x:.4f} psi")
    st.write(f"**Minimum Fuel Consumption:** {y:.6f}")

    # Plot
    P=np.linspace(a,b,400)
    fig,ax=plt.subplots()
    ax.plot(P,f(P),label="f(P)")
    ax.axvline(x,color="r",ls="--",label="P*")
    ax.set_xlabel("Tire Pressure (psi)")
    ax.set_ylabel("Fuel Consumption")
    ax.set_title(alg)
    ax.legend()
    st.pyplot(fig)
