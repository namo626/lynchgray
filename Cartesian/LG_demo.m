%clear all; close all;


%% Input problem data
%==========================================================================
DATA.geometry = 'Cartesian';         % 'Carteisan' or 'Polar'
DATA.geom1    = 60000;                   % x1 or r1
DATA.geom2    = 150000;              % x2 or r2    
DATA.h0       = 3;                   % Bathymetric coefficent
DATA.n        = 0;                   % Bathymetric power
DATA.amp      = 0.3;                 % Tidal amplitude of forcing
DATA.freq     = 2*pi*1/(12.4*3600);  % Tidal frequency of forcing (Note: this frequency corresponds to an M2 tide)
DATA.phase    = 0;                   % Tidal Phase of forcing
DATA.tau      = 0.0025;                   % Linear bottom friction factor
DATA.g        = 9.81;                % Gravitational constant
%--------------------------------------------------------------------------

%% Set up plots
%==========================================================================
h1 = subplot(2,1,1); 
axis(h1,[DATA.geom1,DATA.geom2,-1.2*DATA.amp,1.2*DATA.amp]);
title(h1,'Lynch and Gray Solutions'); 
xlabel(h1,'X'); ylabel(h1,'Surface Elevation');
hold on; box on;

h2 = subplot(2,1,2); hold on; box on;
axis(h2,[DATA.geom1,DATA.geom2,-1/2,1/2]);
xlabel(h2,'X'); ylabel(h2,'Velocity');
hold on; box on;

%% Plot results
%==========================================================================
N = 5; % Number of tidal cycles 
X = linspace(DATA.geom1,DATA.geom2)';
Y = zeros(length(X),1) + 0.087;
coords = [X Y];
T = linspace(0,N*(2*pi*DATA.freq^-1));
for i = 1:length(T)
    cla(h1); cla(h2);
    [SE,VE] = LG2D_Solutions(coords,T(i),DATA);
    plot(h1,X,SE,'b','Linewidth',2)
    plot(h2,X,VE,'r','Linewidth',2)    
    pause(0.10)
    drawnow
end
  