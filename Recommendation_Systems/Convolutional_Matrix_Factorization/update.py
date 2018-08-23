import numpy as np
def update_UV(R,U,V,theta,lambda_u, lambda_v,K):
    num_u=R.shape[0]
    num_v=R.shape[1]
    a=1
    b=0.01
    I_u=np.eye(K)*lambda_u
    I_v=np.eye(K)*lambda_v
    C=np.ones(R.shape)*b
    C[np.where(R>0)]=a
    theta = np.mat(theta.T)
    V=np.mat(V)
    U=np.mat(U)


    print("updating V.......")
    for j in range(num_v):
        # R[:,j](5551,1) U.T(50,5551)
        left=np.dot(np.multiply(U.T,C[:,j]),U)+I_v#(50, 50)
        V.T[:,j]=np.linalg.pinv(left)*(np.dot(np.multiply(U.T,C[:,j]),R[:,j])+lambda_v*theta[:,j])
        #print(j)

    print("updating U.......")
    for i in range(num_u):
        left=np.dot(np.multiply(V.T,C[i,:]),V)+I_u
        #print(left.shape)
        U.T[:,i]=np.linalg.pinv(left)*(np.dot(np.multiply(V.T,C[i,:]),R[i,:].T))
        #print(i)

    E=np.sum(np.multiply(C,np.square(np.dot(U,V.T)-R)))/2+lambda_u*np.sum(np.square(U))/2
    return U,V,E
