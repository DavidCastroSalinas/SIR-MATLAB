%{ 
*** Tarea 2 Programación Avanzada
*** Fecha: 2023-07-11
*** Profesor: Raúl Caulier
*** Alumno: David A. Castro S.
*** DIASMA
%}

close all; %cerramos ventanas de 
clear all; %limpiar consola
clc;

paramLayers.InputLayer = 1;
paramLayers.fullyConnectedLayer=10;
paramLayers.fullyConnectedLayer = 3;

%SGDM "Stochastic Gradient Descent with Momentum"
paramTraining.algoritmoOptimizacion = 'sgdm';
paramTraining.Epocas = 1000;
paramTraining.tasaAprendizajeInicial = 0.01;


%mostrarModeloSIR();
%generarDatosSIR('DATA_SIR.csv');
%ProbarCargaArchivo('DATA_SIR.csv');
ProcesarRedNeuronalSIR('DATA_SIR.csv', paramLayers, paramTraining)


%##########################################################################

function ProbarCargaArchivo(nombreArchivo)    
    % Leer el archivo CSV
    data = readmatrix(nombreArchivo, 'NumHeaderLines', 1);
    t = data(:, 1);
    s = data(:, 2);
    i = data(:, 3);
    r = data(:, 4);

    figure;
    plot(t, s, 'r', 'LineWidth', 2); hold on;
    plot(t, i, 'g', 'LineWidth', 2); hold on;
    plot(t, r, 'b', 'LineWidth', 2); hold on;       
    xlabel('Tiempo');
    ylabel('Población');
    title('Carga Datos de Entrenamiento');
end 

function ProcesarRedNeuronalSIR(nombreArchivo, paramLayers, paramTraining)    
    % Leer el archivo CSV
    data = readmatrix(nombreArchivo, 'NumHeaderLines', 1);
    t = data(:, 1);
    s = data(:, 2);
    i = data(:, 3);
    r = data(:, 4);

    layers = [
        % Capa de entrada TIEMPO
        featureInputLayer(paramLayers.InputLayer)     
        % Capas internas 
        fullyConnectedLayer(paramLayers.fullyConnectedLayer)  
        % Función de activación ReLU
        reluLayer                
        % Capa Salida (SIR 3 neuronas)
        fullyConnectedLayer(paramLayers.fullyConnectedLayer)   
        % Capa de regresión
        regressionLayer];        

    % Configurar las opciones de entrenamiento
    options = trainingOptions( ...
         paramTraining.algoritmoOptimizacion, ...
        'MaxEpochs', paramTraining.Epocas, ...
        'InitialLearnRate', paramTraining.tasaAprendizajeInicial, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
        
    dataTraining = [s i r];
    
    % Entrenar la red neuronal
    net = trainNetwork(t, dataTraining, layers, options);    

    %rango para probar la predicción
    tPred = linspace(-5, 5, 10)';       
         
    % Hacer predicciones con la red entrenada
    sirPrediccion = predict(net, tPred);
    %sPred = sirPrediccion;
    sPred = sirPrediccion(:, 1);
    iPred = sirPrediccion(:, 2);
    rPred = sirPrediccion(:, 3);
    
    figure;
    plot(t, s, 'r', 'LineWidth', 2); hold on;  
    plot(tPred, sPred, 'r--', 'LineWidth', 2);hold on;
    scatter(tPred, sPred, 'r');hold on;  
    legend('Datos Reales', 'Predicciones de la Red');
    xlabel('Tiempo');ylabel('Población');title('Red Neuronal');

    figure;
    plot(t, i, 'g', 'LineWidth', 2); hold on;
    plot(tPred, iPred, 'g--', 'LineWidth', 2);hold on;
    scatter(tPred, iPred, 'g');hold on;   
    legend('Datos Reales', 'Predicciones de la Red');
    xlabel('Tiempo');ylabel('Población');title('Red Neuronal')

    figure;
    plot(t, r, 'b', 'LineWidth', 2); hold on;
    plot(tPred, rPred, 'b--', 'LineWidth', 2);hold on;
    scatter(tPred, rPred, 'b');hold on;
    legend('Datos Reales', 'Predicciones de la Red');
    xlabel('Tiempo');ylabel('Población');title('Red Neuronal')

    %######
    figure;
    plot(t, s, 'r', 'LineWidth', 2); hold on;
    plot(t, i, 'g', 'LineWidth', 2); hold on;
    plot(t, r, 'b', 'LineWidth', 2); hold on;
    
    plot(tPred, sPred, 'r--', 'LineWidth', 2);hold on;
    plot(tPred, iPred, 'g--', 'LineWidth', 2);hold on;
    plot(tPred, rPred, 'b--', 'LineWidth', 2);hold on;

    scatter(tPred, sPred, 'r');hold on;
    scatter(tPred, iPred, 'g');hold on;
    scatter(tPred, rPred, 'b');hold on;
    
    legend('Datos Reales', 'Predicciones de la Red');
    xlabel('Tiempo');
    ylabel('Población');
    title('Red Neuronal');
end


function generarDatosSIR(nombreArchivo)
    % Parámetros
    b = 1.2; %tasa transmisión de la enfermedad
    c = 0.2; %tasa de de infectados al de retirados 
    
    % Condiciones iniciales
    susceptibles0 = 9.5;
    infectados0 = 0.5;
    recuperados0 = 0;

    sir0 = [susceptibles0, infectados0, recuperados0];

    % Vector de tiempo
    t = linspace(-5, 5, 1000)';
    
    % sistema de ecuaciones diferenciales
    [t, sol] = ode45(@(t, data) ...
        calculoSIRCondiciones(data, b, c), t, sir0);
        
    s = sol(:, 1);
    i = sol(:, 2);
    r = sol(:, 3);

    T = table(t, s, i, r, 'VariableNames', {'Tiempo', ...
        'Susceptibles', 'Infectados', 'Recuperados'});
    writetable(T, nombreArchivo);    
    print("Archivo generado correctamente");
end

% Definir la función que describe el sistema de ecuaciones diferenciales
function dydt = calculoSIRCondiciones(y, b, c)
    S = y(1);
    I = y(2);
    R = y(3);
    dSdt = -b * I * S / (S + I + R);
    dIdt = b * I * S / (S + I + R) - c * I;
    dRdt = c * I;
    dydt = [dSdt; dIdt; dRdt];
end



function mostrarModeloSIR() 
    % Definir los parámetros
    b = 1.2; %tasa per-cápita de transmisión de la enfermedad
    c = 0.2; %El flujo de paso del compartimento de infectados al de retirados 
    
    % Condiciones iniciales
    y0 = [950, 50, 0];
    
    % Vector de tiempo
    t = linspace(0, 10, 1001);
    y = [0,0,0];
    
    % Resolver el sistema de ecuaciones diferenciales
    [t, sol] = ode45(@(t, y) calculoSIRCondiciones(y, b, c), t, y0);
   
    % Graficar los resultados
    figure;
    plot(t, sol(:, 1), 'r', 'LineWidth', 2); hold on;
    plot(t, sol(:, 2), 'g', 'LineWidth', 2); hold on;
    plot(t, sol(:, 3), 'b', 'LineWidth', 2);
    legend('S', 'I', 'R');
    xlabel('Tiempo');
    ylabel('Población');
    title('Modelo SIR');
end

