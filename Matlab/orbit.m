classdef orbit < handle
    %{
    Defines the 'orbit' object.
    Initialised with either state vectors or classical orbital elements.
    By default, a circular equatorial orbit with semi-major axis of 7000km is initialised.
    
    Angular variables must be in degrees [°].
    Units must be consistent with mu, which by default is in [km] and [s].
    %}
    
    properties (SetObservable,AbortSet)
        a %semi-major axis (default: 7000 [km])
        e %eccentricity (must be >0)
        i %inclination [°]
        Omega %RAAN, or longitude of right ascending node [°]
        omega %argument of periapsis [°]
        f %true anomaly [°]
        R %position vector (as a list or numpy array)
        V %velocity vector (as a list or numpy array)
        P %Orbital period, can be used to describe the orbit size; has priority over 'a'
    end
    
    properties (Hidden)
        update_elements = false %Whether to update the other elements after a new assignment
        update_message = true %Whether to output a message when updating elements
    end
    
    properties (SetAccess = immutable)
        mu %gravitational parameter (default: 398600 [km^3/s^2] (mu for Earth))
    end
    
    properties (Dependent)
        E %Eccentric anomaly
        M %Mean anomaly
        ap %Apoapsis
        pe %periapsis
    end
    
    methods
        function self = orbit(args)     %Constructor
            arguments
                args.a = 7000
                args.e = 0
                args.i = 0
                args.Omega = 0
                args.omega = 0
                args.f = 0
                args.R
                args.V
                args.P
                args.mu = 398600
            end
            
            self.mu = args.mu;
            if isfield(args,'R') && isfield(args,'V') 
                self.R=args.R;
                self.V=args.V;
                [self.a,self.e,self.i,self.Omega,self.omega,self.f]=space.calc_coe(args.R,args.V,args.mu);
                self.P=2*pi*(self.a^3/args.mu)^(1/2);
            elseif isfield(args,'R') || isfield(args,'V')
                throw(MException("orbit:InitError:RV","R or V undefined."))
%                 raise InitError("R or V undefined.")
            else
                if isfield(args,'P')
                    self.P=args.P;                                %[s]
                    self.a=((P/(2*pi))^2*args.mu)^(1/3);
                else
                    self.a=args.a;
                    self.P=2*pi*(self.a^3/args.mu)^(1/2);    %[s]
                end
                self.e=args.e;
                self.i=args.i;
                self.Omega=args.Omega;
                self.omega=args.omega;
                self.f=args.f;
                [self.R,self.V]=space.calc_rv(args.a,args.e,args.i,args.Omega,args.omega,args.f,args.mu);
            end
            self.regulate();
            self.update_elements = true;
            addlistener(self,{'a','e','i','Omega','omega','f','P','R','V'},'PostSet',@self.specset);
        end
        function specset(self,prop,~)
%             Special set method for the elements that require updating
%             other elements.
            name = prop.Name;
            if self.update_elements
                self.update_elements = false;
                self.regulate();
                if strcmp(name,'P')
                    self.a=((self.P/(2*pi))^2*self.mu)^(1/3);
                    if self.update_message; disp('Updated a'); end
                end
                if ismember(name,{'a','e','i','Omega','omega','f','P'})
                    [self.R,self.V]=space().calc_rv(self.a,self.e,self.i,self.Omega,self.omega,self.f,self.mu);
                    if self.update_message; disp('Updated R and V'); end
                end
                if ismember(name,{'R','V'})
                    [self.a,self.e,self.i,self.Omega,self.omega,self.f]=space().calc_coe(self.R,self.V,self.mu);
                    self.P=2*pi*(self.a^3/self.mu)^(1/2);
                    if self.update_message; disp('Updated COE'); end
                end
                if strcmp(name,'a')
                    self.P=2*pi*(self.a^3/self.mu)^(1/2);
                    if self.update_message; disp('Updated P'); end
                end
                self.update_elements = true;
            end
        end
        function [a,e,i,Omega,omega,f] = coe(self)
%             Returns classical orbital parameters
            a=self.a;
            e=self.e;
            i=self.i;
            Omega=self.Omega;
            omega=self.omega;
            f=self.f;
        end
        function [R,V] = rv(self)
%             Returns state vectors
            R=self.R;
            V=self.V;
        end
        function copied = copy(self)
%             Returns copy of itself
            copied = orbit(a=self.a,e=self.e,i=self.i,Omega=self.Omega,omega=self.omega,f=self.f,mu=self.mu);
            copied.update_elements = self.update_elements;
        end
        function self = update(self,args)
%             Serves to change some parameters of the orbit.
%             Does not allow to change the value for mu, as it would be ambiguous whether to keep the
%             classical orbital elements or the state vectors. It is better to use a new 'orbit' instance.
%             'P' has again priority over 'a'
%             Classical orbital elements have priority over state vectors
%             Returns the updated 'orbit'
            arguments
                self;
                args.a
                args.e
                args.i
                args.Omega
                args.omega
                args.f
                args.R
                args.V
                args.P
            end
            restore_update_elements = self.update_elements;
            self.update_elements = false;
            if isfield(args,'P'); args.a=((args.P/(2*pi))^2*self.mu)^(1/3); end
            keys1={'a','e','i','Omega','omega','f'};
%             keys1=container.Map({'a','e','i','Omega','omega','f'},{args.a,args.e,args.i,args.Omega,args.omega,args.})
%             {'a':a,'e':e,'i':i,'Omega':Omega,'omega':omega,'f':f}
            keys2={'R','V'};
            if any(arrayfun(@(key) isfield(args,key),keys1))
                for x = keys1; if isfield(args,x); eval(['self.',x{1},'=getfield(args,''',x{1},''');']); end; end
                self.regulate();
                [self.R,self.V]=space.calc_rv(self.a,self.e,self.i,self.Omega,self.omega,self.f,self.mu);
                self.P=2*pi*(self.a^3/self.mu)^(1/2);
            elseif any(arrayfun(@(key) isfield(args,key),keys2))
                for x = keys2
                    if isfield(args,x)
                        eval(['self.',x{1},'=getfield(args,''',x{1},''');']);
                    end
                end
                [self.a,self.e,self.i,self.Omega,self.omega,self.f]=space.calc_coe(self.R,self.V,self.mu);
                self.P=2*pi*(self.a^3/self.mu)^(1/2);
            end
            self.update_elements = restore_update_elements;
        end
        function E = get.E(self)
            %{
            Returns the eccentric anomaly
            %}
            e=self.e; %#ok<PROP>
            f=deg2rad(self.f); %#ok<PROP>
            E=acos((e+cos(f))/(1+e*cos(f))); %#ok<PROP>
            if self.f>pi
                E=2*pi-E;
            end
            E = rad2deg(E);
        end
        function M = get.M(self)
            %{
            Returns the mean anomaly
            %}
            e=self.e; %#ok<PROP>
            E=deg2rad(self.E); %#ok<PROP>
            M = rad2deg(E-e*sin(E)); %#ok<PROP>
        end
        function ap = get.ap(self)
            %{
            Returns the apoapsis
            %}
            ap = self.a*(1+self.e);
        end
        function pe = get.pe(self)
            %{
            Returns the periapsis
            %}
            pe = self.a*(1-self.e);
        end
    end
    
    methods%(Access=protected)
        function self = regulate(self)
%             Internal method used to keep the values of the object's parameters within normal bounds.
        if self.e<0
            throw(MException('orbit.regulate:ExcentricityAssertion',"Invalid excentricity: %s",self.e))
        end
%         assert self.e<1, "Orbit is not closed; excentricity > 1"
%         self.i=mod((self.i+90),180)-90
        if self.i<0; self.Omega=self.Omega+180; end
        self.i=rad2deg(acos(cos(deg2rad(self.i))));    %So that 180° is valid
        self.Omega=mod(self.Omega,360);
        self.omega=mod(self.omega,360);
        self.f=mod(self.f,360);
        end
    end
end
