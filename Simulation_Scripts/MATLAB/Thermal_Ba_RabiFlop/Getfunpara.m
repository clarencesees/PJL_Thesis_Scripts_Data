function [BETA,BETA_Err,Exitflag]=Getfunpara(varargin)
%[BETA,BETA_Err,Exitflag]=Getfunpara(fun,P0,x,y,P_LB,P_UB,options)
%
%Input definition:
%fun: Function for nonlinear regression.
%P0: Initial guess of parameters.
%x: x data points.
%y: y data points.
%P_LB: Lower bound of parameters.
%P_UB: Upper bound of parameters.
%options: options for nonlinear regression.
%
%Output definition:
%BETA: Converged parameters.
%BETA_Err: Uncertainties of converged parameters.
%Exitflag: Condition that lead to termination of regression.

if nargin==4
    fun=varargin{1};
    P0=varargin{2};
    x=varargin{3};
    y=varargin{4};
    
    [BETA,~,R,Exitflag,~,~,J]=lsqcurvefit(fun,P0,x,y);
    MSE=(R(:)'*R(:))/(numel(y)-numel(P0));
    COVB = pinv(full(J'*J))*MSE;
    BETA_Err=sqrt(spdiags(COVB,0));
elseif nargin==6
    fun=varargin{1};
    P0=varargin{2};
    x=varargin{3};
    y=varargin{4};
    P_LB=varargin{5};
    P_UB=varargin{6};
    
    [BETA,~,R,Exitflag,~,~,J]=lsqcurvefit(fun,P0,x,y,P_LB,P_UB);
    MSE=(R(:)'*R(:))/(numel(y)-numel(P0));
    COVB = pinv(full(J'*J))*MSE;
    BETA_Err=sqrt(spdiags(COVB,0));
elseif nargin==7
    fun=varargin{1};
    P0=varargin{2};
    x=varargin{3};
    y=varargin{4};
    P_LB=varargin{5};
    P_UB=varargin{6};
    options=varargin{7};
    
    [BETA,~,R,Exitflag,~,~,J]=lsqcurvefit(fun,P0,x,y,P_LB,P_UB,options);
    MSE=(R(:)'*R(:))/(numel(y)-numel(P0));
    COVB = pinv(full(J'*J))*MSE;
    BETA_Err=sqrt(spdiags(COVB,0));
end