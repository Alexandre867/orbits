%{
Copyright (c) 2022 Alexandre Daigneault

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
%}

classdef space
    methods(Static)%,Access=protected)
        function A = clip(A,args)
            arguments
                A
                args.minv
                args.maxv
            end
            if isfield(args,'minv')
                A(A<args.minv)=args.minv;
            end
            if isfield(args,'maxv')
                A(A>args.maxv)=args.maxv;
            end
        end
    end
    methods(Static)
        function angle = ang_bet(A,B)
%             Angle between vectors A and B. (Vectors as numpy arrays)
            angle=acos(space.clip(dot(A,B)/(norm(A)*norm(B)), minv=-1.0, maxv=1.0));
        end
        function [a,e,i,Omega,omega,f] = calc_coe(R,V,mu)
%             Returns the classical orbital elements for state vectors 'R' and 'V' and gravitational parameter 'mu'.
        
%             arguments
%                 args.R
%                 args.V
%                 args.mu
%             end
%             R = args.R;
%             V = args.V;
%             mu = args.mu;

            r=norm(R);
            v=norm(V);
            epsilon = v^2/2-mu/r;
            a=-mu/(2*epsilon);
%             disp(R)
%             disp(V)
%             disp(mu)
            E=1/mu*((v^2-mu/r)*R-(R*V')*V);
            e=norm(E);
            H=cross(R,V);         %vector of h
%             h=norm(H)
            i=rad2deg(space.ang_bet([0,0,1],H));
            N=cross([0,0,1],H);   %vector of n
            if norm(N)==0
                N=[1,0,0];
            end
    %         n=norm(N)
            Omega=rad2deg(space.ang_bet([1,0,0],N));
            if N(2)<0
                Omega=mod((360-Omega),360);
            end
            if e==0
                omega=0;
                E=N;
            else
                omega=rad2deg(space.ang_bet(N,E));
%                 if E(3)<0: omega=360-omega            %%%%%% NOT WORKING FOR i=0
%                 if sum(E.*cross([0,0,1],N))<0: omega=360-omega  %%%%% NOT WORKING FOR i=180
                if sum(E.*cross(H,N))<0
                    omega=360-omega;
                end
            end
            f=rad2deg(space.ang_bet(E,R));
            if sum(R.*V)<0
                f=360-f;
            end
        end
        function [R,V] = calc_rv(a,e,i,Omega,omega,f,mu)
%             Returns the state vectors 'R' and 'V' from the classical orbital elements and the gravitational parameter 'mu'.
%             f can be value or horizontal array of values
            i=deg2rad(i);
            Omega=deg2rad(Omega);
            omega=deg2rad(omega);
            f=deg2rad(f);
            T_i=[[1,0,0];
                 [0,cos(i),-sin(i)];
                 [0,sin(i),cos(i)]];
            T_Omega=[[cos(Omega),-sin(Omega),0];
                     [sin(Omega),cos(Omega),0];
                     [0,0,1]];
            T_mat=T_Omega*T_i;
            r=a*(1-e^2)./(1+e*cos(f));
            v=sqrt(mu*(2./r-1/a));
            l=omega+f;
            R=T_mat*[cos(l);sin(l);zeros(1,length(l))].*r;
            R=R';
            rp=a*(1-e^2)./(1+e*cos(f)).^2*e.*sin(f);    %dr/df
            V=[cos(l).*rp-sin(l).*r;sin(l).*rp+cos(l).*r;zeros(1,length(l))]';
            V=V./sqrt(sum(V.^2,2)).*v';
            V=T_mat*V';
            V=V';
        end
        function varargout = predict(orb,t,rv)
%             Returns position (if 'rv' is True) of an object after a given time interval.
%             For 'rv' is False, returns the true anomaly.
            arguments
                orb orbit
                t {mustBeNumeric}
                rv = true
            end
            
            t=mod(t,orb.P);
            e=orb.e;
%             f=deg2rad(orb.f)
%             E=acos((e+cos(f))/(1+e*cos(f)))
%             if f>pi; E=2*pi-E; end
%             M=E-e*sin(E)
            M=deg2rad(orb.M);
            n=2*pi/orb.P;
            M=mod((M+n*t),(2*pi));
            E=M;
            iterations = 6;
            for i = 1:iterations; E=E-(E-e*sin(E)-M)./(1-e*cos(E)); end    % Newton's method
            f=acos((cos(E)-e)./(1-e*cos(E)));
            if E>pi; f=2*pi-f; end
            f=rad2deg(f);
            if rv
                if length(t)-1; [varargout{1:2}]=arrayfun(@(f) orb.copy().update(f=f).rv(),f,'UniformOutput',false); %#ok<BDLOG>
                else; [varargout{1:2}]=orb.copy().update(f=f).rv();
                end
            else; varargout{1} = f;
            end
        end
        function [lon,lat,r] = ground_track_pos(orb,t,t_0,args)
%             Returns the ground track for a given orbit and a given time interval as lists of longitude, latitude and distance R.
%             Assumes that (0°N, 0°W) is aligned with [1,0,0] of the orbit's reference frame at t=0.
%             
%             orbit: orbit to track
%             t: end of time range (units depending on mu of orbit, default: [s])
%             t_0: beginning of time range (default: 0)
%             dt: step interval for the reported positions; doesn't affect accuracy (default: (t-t_0)/1000)
%             time: if True (default), t, t_0 and dt are given in time. Otherwise, they are taken as defining the range to report in degrees of true anomaly, with t_0=0 being the current position of the orbit object.

            arguments
                orb orbit
                t {mustBeNumeric}
                t_0 {mustBeNumeric} = 0
                args.dt {mustBeNumeric} = (t-t_0)/1000
                args.time=true
            end
%             if dt is None: dt=(t-t_0)/1000
            dt = args.dt;
            time = args.time;
            
            if ~(time)
%                 throw(MException("space:ground_track_posError:time","time=false is not implemented."))
                T=orb.f+t_0:dt:orb.f+t+t_0; 
                o2=orb.copy();
%                 pos_cart=np.vstack(list(map(lambda x: o2.update(f=x).R,T)))
                if length(t)-1 %#ok<BDLOG>
                    pos_cart=arrayfun(@(f) o2.update(f=f).rv(),T,'UniformOutput',false);
                else; pos_cart=o2.update(f=f).rv();
                end
                T = idivide(mod(arrayfun(@(x) o2.update(f=x).M,T)-orbit.M,360)/(360/orbit.P)+(T-orbit.f),360)*orbit.P;
            else
                T=t_0:dt:t;
%                 pos_cart=np.vstack(list(map(lambda x: self.predict(orbit,x,rv=True)[0],T)))
                pos_cart=space.predict(orb,T,true);
            end
            pos_cart=reshape([pos_cart{:}],3,[])';
%             display(pos_cart)
%             display(T)

            lon = mod(rad2deg(atan2(pos_cart(:,2),pos_cart(:,1)))'-T*360/86164.0905+180,360)-180;
%             lon=lon-lon(0)
            lat = 90-rad2deg(atan2((pos_cart(:,1).^2+pos_cart(:,2).^2).^(1/2),pos_cart(:,3)))';
            r = (pos_cart(:,1).^2+pos_cart(:,2).^2+pos_cart(:,3).^2)'.^(1/2);
%             return lon,lat,r
        end
        function plot3d_orbit(orbs,args)
            arguments
                orbs %One or several orbits to plot in array []
                args.R = 6371 %Planet radius [km] (default: Earth radius)
                args.res = 100 %Resolution of the orbit calculation
                args.faces = 20 %Number of faces for planet
                args.planet_color = [0,.25,.5] %Color of the planet
            end
            
            f=0:360/(args.res-1):360;
            
            [X,Y,Z]=sphere(args.faces);
            hold off
            surf(X*args.R,Y*args.R,Z*args.R,repmat(reshape(args.planet_color,1,1,[]),size(X,1),size(X,2)))
            hold on
            axis equal
            
            for orb = orbs
                [R,~]=space.calc_rv(orb.a,orb.e,orb.i,orb.Omega,orb.omega,f,orb.mu);
                plot3(R(:,1),R(:,2),R(:,3),'LineWidth',2)
%                 comet3(R(:,1),R(:,2),R(:,3),.1)
                [R,~]=space.calc_rv(orb.a,orb.e,orb.i,orb.Omega,orb.omega,0,orb.mu);
                plot3(R(:,1),R(:,2)',R(:,3),'s','MarkerSize',10,'LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','white')
                [R,~]=space.calc_rv(orb.a,orb.e,orb.i,orb.Omega,orb.omega,180,orb.mu);
                plot3(R(:,1),R(:,2)',R(:,3),'s','MarkerSize',10,'LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','white')
                [R,~]=space.calc_rv(orb.a,orb.e,orb.i,orb.Omega,orb.omega,orb.f,orb.mu);
                plot3(R(:,1),R(:,2)',R(:,3),'o','MarkerSize',10,'LineWidth',2,'MarkerEdgeColor','b')%,'MarkerFaceColor','white')
            end
            axis equal
        end
                
    end
end
