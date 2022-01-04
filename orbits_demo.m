o1 = orbit(a=1e4,e=.3,i=40,Omega=0,omega=30,f=120)
o2 = orbit(a=1e4,e=.2,i=20,Omega=90,omega=180,f=230)
space().plot3d_orbit([o1,o2])