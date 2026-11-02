function [SE,Ubar,Vbar] = LG2D_Solutions(coords,t,DATA)

%-----------------------------------------------------------------------
%
%  [SE,Ubar,Vbar] = LG2D_Solutions( coord,t,DATA )
%
%  This function computes the surface elevation, SE, and horizontal and
%  depth-averaged velocity, Ubar, for the test cases for the 2D depth-
%  averagedlinear shallow water equations presented in [1] at the point
%  (coord, t).
%
%-----------------------------------------------------------------------
%
%  Input:
%  ------
%
%    coords:    x and y coordinates at which to compute the solution
%    t:         The time at which to compute the solution
%    DATA:      Data structure of the problem parameters
%      |
%      |-- geometry:  The string 'Cartesian' or 'Polar'
%      |-- geom1:     x1 or r1
%      |-- geom2:     x2 or r2
%      |-- h0:        Bathymetric coefficent
%      |-- n:         Bathymetric power
%      |-- amp:       Tidal amplitude of forcing
%      |-- freq:      Tidal frequency of forcing
%      |-- phase:     Tidal Phase of forcing
%      |-- tau:       Linear bottom friction factor
%      |-- g:         Gravitational constant
%
%  Output:
%  -------
%
%   SE:     The surface elevation at the point coord at time t
%   Ubar:   The velocity in the x direction at the spatial coordinates
%           defined in the input array coord at time t.
%   Vbar:   The velocity in the y direction at the spatial coordinates
%           defined in the input array coord at time t.
%
%-----------------------------------------------------------------------
%
%  Note: Reference [2] provides corrected solutions of the reverse 
%        quadratic bathymetry cases (n = -2) as the original solutions 
%        published in [1] are in error.
%
%-----------------------------------------------------------------------
% 
%  References:  
%
%     [1]  Lynch, Daniel R and Gray, William G
%          Analytic Solutions for Computer Flow Model Testing  
%          Journal of the Hydraulics Division
%          Vol. 104 (HY10), 1978, 1409-1428.
%
%     [2]  Chen, Ching L.
%          Analytic Solutions for Tidal Model Testing
%          Journal of Hydraulic Engineering
%          Vol. 115 (12), 1989, 1707-1714.
%
%-----------------------------------------------------------------------
%
%  Written by Ethan Kubatko
%
%-----------------------------------------------------------------------

% Compute the constant beta and the product of i, freq, and t 

DATA.amp = DATA.amp*exp(-1i*DATA.phase);
beta = sqrt((DATA.freq^2 - 1i*DATA.freq*DATA.tau)/(DATA.g*DATA.h0));
iwt = 1i*DATA.freq*t; 

switch DATA.geometry
    case {'Polar','polar','radial'}
        %% ----------------------------------------------------------------
        % Radial coordinate test cases
        %------------------------------------------------------------------
        r1 = DATA.geom1;
        r2 = DATA.geom2;
        r  = sqrt(coords(:,1).^2 + coords(:,2).^2);
        switch DATA.n
            case  0
                % Compute the constants                
                A =  DATA.amp*bessely(1,beta*r1)/(besselj(0,beta*r2)*bessely(1,beta*r1) - besselj(1,beta*r1)*bessely(0,beta*r2));
                B = -DATA.amp*besselj(1,beta*r1)/(besselj(0,beta*r2)*bessely(1,beta*r1) - besselj(1,beta*r1)*bessely(0,beta*r2));                
                % Compute the solutions                
                SE = real((  A*besselj(0,beta*r) + B*bessely(0,beta*r))*exp(iwt));
                Ur = real(( -A*besselj(1,beta*r) - B*bessely(1,beta*r))*1i*DATA.freq/(beta*DATA.h0)*exp(iwt));
            case  1
                % Compute the constants                
                A =  DATA.amp*sqrt(r2)*bessely(2,2*beta*sqrt(r1))/(besselj(1,2*beta*sqrt(r2))*bessely(2,2*beta*sqrt(r1)) - besselj(2,2*beta*sqrt(r1))*bessely(1,2*beta*sqrt(r2)));
                B = -DATA.amp*sqrt(r2)*besselj(2,2*beta*sqrt(r1))/(besselj(1,2*beta*sqrt(r2))*bessely(2,2*beta*sqrt(r1)) - besselj(2,2*beta*sqrt(r1))*bessely(1,2*beta*sqrt(r2)));                
                % Compute the solutions                
                SE = real(1/sqrt(r)*(A*besselj(1,2*beta*sqrt(r)) + B*bessely(1,2*beta*sqrt(r)))*exp(iwt));
                Ur = real(1/r*(-A*besselj(2,2*beta*sqrt(r)) - B*bessely(2,2*beta*sqrt(r)))*1i*DATA.freq/(beta*DATA.h0)*exp(iwt));
            case  2
                % Compute the constants                
                s1 = -1 + sqrt(1-beta^2);
                s2 = -1 - sqrt(1-beta^2);                
                A  =  DATA.amp*s2*r1^s2/(s2*r2^s1*r1^s2 - s1*r1^s1*r2^s2);
                B  = -DATA.amp*s1*r1^s1/(s2*r2^s1*r1^s2 - s1*r1^s1*r2^s2);                
                % Compute the solutions                
                SE = real((A*r.^s1 + B*r.^s2)*exp(iwt));
                Ur = real((s1*A*r.^(s1-1) + s2*B*r.^(s2-1))*1i*DATA.freq/(beta^2*DATA.h0)*exp(iwt));
            case -2
                % Compute the constants                
                s1 = -1 + sqrt(1-beta^2);
                s2 = -1 - sqrt(1-beta^2);                
                A  =  DATA.amp*s2*r1^s2/(s2*r2^s1*r1^s2 - s1*r1^s1*r2^s2);
                B  = -DATA.amp*s1*r1^s1/(s2*r2^s1*r1^s2 - s1*r1^s1*r2^s2);                
                % Compute the solutions                
                SE = real(DATA.amp*(cos(beta/2*(r^2-r1^2))/cos(beta/2*(r2^2-r1^2))*exp(iwt)));
                Ur = real(DATA.amp*r*(sin(beta/2*(r^2-r1^2))/cos(beta/2*(r2^2-r1^2))*1i*DATA.freq/(beta*DATA.h0)*exp(iwt)));
        end
        theta = atand(coords(:,2)./coords(:,1));
        Ubar = Ur.*cosd(theta);
        Vbar = Ur.*sind(theta);
    case {'Cartesian','cartesian','Cart.'}
        %% ----------------------------------------------------------------
        % Cartesian coordinate test cases
        %------------------------------------------------------------------
        x1 = DATA.geom1;
        x2 = DATA.geom2;
        x  = coords(:,1);        
        switch DATA.n
            case  0
                % Compute the solutions                
                SE = real(DATA.amp*exp(iwt)*cos(beta*(x-x1))/cos(beta*(x2-x1)));
                Ubar = real(-1i*DATA.freq*DATA.amp/(beta*DATA.h0)*exp(iwt)*sin(beta*(x-x1))/cos(beta*(x2-x1)));
            case  1
                % Compute the constants                
                A =  DATA.amp*bessely(1,2*beta*sqrt(x1))/(besselj(0,2*beta*sqrt(x2))*bessely(1,2*beta*sqrt(x1)) - bessely(0,2*beta*sqrt(x2))*besselj(1,2*beta*sqrt(x1)));
                B = -DATA.amp*besselj(1,2*beta*sqrt(x1))/(besselj(0,2*beta*sqrt(x2))*bessely(1,2*beta*sqrt(x1)) - bessely(0,2*beta*sqrt(x2))*besselj(1,2*beta*sqrt(x1)));                
                % Compute the solutions                
                SE   = real((A*besselj(0,2*beta*sqrt(x)) + B*bessely(0,2*beta*sqrt(x)))*exp(iwt));
                Ubar = real(1/sqrt(x)*(-A*besselj(1,2*beta*sqrt(x)) - B*bessely(1,2*beta*sqrt(x)))*1i*DATA.freq/(beta*DATA.h0)*exp(iwt));
            case  2
                % Compute the constants                
                s1 = -1/2 + sqrt(1/4 - beta^2);
                s2 = -1/2 - sqrt(1/4 - beta^2);                
                A  =  DATA.amp*s2*x1^s2/(s2*x1^s2*x2^s1 - s1*x1^s1*x2^s2);
                B  = -DATA.amp*s1*x1^s1/(s2*x1^s2*x2^s1 - s1*x1^s1*x2^s2);                
                % Compute the solutions                
                SE   = real((A*x.^s1 + B*x.^s2)*exp(iwt));
                Ubar = real((A*s1*x.^(s1-1) + B*s2*x.^(s2-1))*1i*DATA.freq/(beta^2*DATA.h0)*exp(iwt));
            case -2
                % Compute the constants                
                A = DATA.amp*besselj( 1/4,beta*x1^2/2)/(x2^(3/2)*(besselj(1/4,beta*x1^2/2)*besselj(3/4,beta*x2^2/2) + besselj(-3/4,beta*x2^2/2)*besselj(-1/4,beta*x1^2/2)));
                B = DATA.amp*besselj(-1/4,beta*x1^2/2)/(x2^(3/2)*(besselj(1/4,beta*x1^2/2)*besselj(3/4,beta*x2^2/2) + besselj(-3/4,beta*x2^2/2)*besselj(-1/4,beta*x1^2/2)));                
                % Compute the solutions                
                SE   = real(x^(3/2)*(A*besselj( 3/4,beta*x^2/2) + B*besselj(-3/4,beta*x^2/2))*exp(iwt));
                Ubar = real(x^(5/2)*(A*besselj(-1/4,beta*x^2/2) - B*besselj( 1/4,beta*x^2/2))*1i*DATA.freq/(beta*DATA.h0)*exp(iwt));
        end
        Vbar = zeros(size(Ubar));
        
end
end