
import numpy as np
from sklearn.decomposition import SparseCoder
# import librosa

class DDSC():
    def __init__(self, train_set, train_sum, alpha, 
                 epsilon, reg_lambda, steps, n, m, T, k):
        """
        Inputs:
            train_set: dict of X_i matrix with dim T*m for each individual appliance i 
            train_sum: dataframe of X_sum aggregated matrix T*m 
            alpha: gradiant rate for the convergence step for DD (4b).
            epsilon: gradient stepsize of the pre-training (2e) ||A_t+1 - A_t||< epsilon 
            reg_lambda: reguarization weight of penalty function
            steps: interations to be performed for the convergence part
            n: number of basis functions 
            m: number of features (households)
            T: number of samples (hours)
            k: number of applicances i (1, k)
        """
        self.train_set = train_set.values()
        self.train_sum = train_sum.values
        self.alpha = alpha 
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.steps = steps
        self.n = n 
        self.m = m
        self.T = T
        self.k = k
        
        # ======= Instances that can be used for plotting =====
        self.acc_nnsc = None
        self.err_nnsc = None
        self.acc_ddsc = None
        self.err_ddsc = None
        
        self.a_nnsc = None
        self.b_nnsc = None
        self.a_ddsc = None
        self.b_ddsc = None

    def _initialization(self):
        '''
        DDSC step 1
        initiualize the matrices A,B with positive values
        scale columns of B s.t b(j) = 1
        '''
        A = np.random.random((self.n,self.m)) # A: n*m
        B = np.random.random((self.T,self.n)) # B: T*n

        # scale columns s.t. b_i^(j) = 1
        B /= sum(B) 
        
        return A, B
    
    @staticmethod
    def _pos_constraint(mat):
        '''
        nnsc step 2(b)
        using only the positive values of matrix  
        input: matrix n*m 
        '''     
        indices = np.where(mat < 0.0)
        mat[indices] = 0.0
        return mat   
    
    def nnsc(self):
        '''
        Method as in NNSC from nonnegative sparse coding finland.
        from P.Hoyer

        return:
            A_list, B_list: list of A and B for each appliance i 
        '''
        
        acc_nnsc = []
        err_nnsc = []
        a_nnsc = []
        b_nnsc = []
        
        # used for F
        X_train = self.train_set # dict_value 
        A_list = []
        B_list = []
        
        for X in X_train:
            # step 1 
            A0, B0 = self._initialization() # initialization 
            Ap, Bp = A0, B0 
            Ap1, Bp1 = Ap, Bp # record previous step Ap, Bp
            t = 0
            change_A = 1.0
            while t <= self.steps and change_A >= self.epsilon:            
                Bp = Bp - self.alpha * np.dot((np.dot(Bp, Ap) - X), Ap.T) # step 2a
                Bp = self._pos_constraint(Bp) # step 2b 
                Bp /= sum(Bp) # step 2c 
                
                # step 2d
                dot_part2 = np.divide(np.dot(Bp.T, X), (np.dot(np.dot(Bp.T, Bp), Ap) + self.reg_lambda)) # element wise division 
                Ap = np.multiply(Ap, dot_part2)

                change_A = np.linalg.norm(Ap - Ap1)
                change_B = np.linalg.norm(Bp - Bp1)
                Ap1, Bp1 = Ap, Bp
                t += 1
                
                if t % 10 == 0:
                    print("iter {t}：A change = {a:8.4f}".format(t=t, a=change_A))
                
            print("Gone through one appliance.\n")
            A_list.append(Ap)
            B_list.append(Bp)


        # for thesis
        acc_iter = self.accuracy(X_train, self.train_sum, B_list, A_list)
        err_iter = self.error(X_train, self.train_sum, B_list, A_list)
        acc_nnsc.append(acc_iter)
        err_nnsc.append(err_iter)
        # append norm of matrices
        a_nnsc.append(np.linalg.norm(sum(A_list)))
        b_nnsc.append(np.linalg.norm(sum(B_list)))

        self.acc_nnsc = acc_nnsc
        self.err_nnsc = err_nnsc
        self.a_nnsc = a_nnsc
        self.b_nnsc = b_nnsc
        
        return A_list, B_list

    def accuracy(self, X_train, X_sum, B, A):
        '''
        inputs:
            X_train: dict_value of list 
        
        Everything needs to be in lists of ndarrays
        of the components
        '''
        B_cat = np.hstack(B)
        A_cat = np.vstack(A)

        A_prime = self.F(X_sum, B_cat, A=A_cat)
        A_last = np.split(A_prime, self.k, axis=0)
        X_predict = self.predict(A_last, B)
        
        
        X_train = list(X_train)
        

        acc_numerator = [np.sum(np.minimum((B[i].dot(A_last[i])).sum(axis=0), (sum(X_train[i].sum(axis=0)))))
                         for i in range(len(B))]
        
        
        acc_denominator = sum(X_predict).sum()
        acc = sum(acc_numerator) / acc_denominator
        
        acc_denominator = X_sum.sum()
        acc_star = sum(acc_numerator) / acc_denominator
        return acc, acc_star

    def get_accuracy_plot(self):
        return self.acc_nnsc, self.acc_ddsc

    def get_error_plot(self):
        return self.err_nnsc, self.err_ddsc

    def get_a(self):
        return self.a_nnsc, self.a_ddsc

    def get_b(self):
        return self.b_nnsc, self.b_ddsc

    def error(self,X, X_sum, B, A):
        '''
        Error for the whole disaggregation part within list, sum the list to get
        the resulting disaggregation
        Parameters : must have x_train as x
        '''
        B_cat = np.hstack(B)
        A_cat = np.vstack(A)
        
        
        error = [(1.0/2.0) * np.linalg.norm((list(X)[i] - B[i].dot(A[i]))**2) for i in range(self.k)]
        error = sum(error)
        
        A_last_error = self.F(X_sum, B_cat,A_cat)
        
        A_last_error_list = np.split(A_last_error,self.k,axis=0)
        error_star = [(1.0/2.0) * np.linalg.norm((list(X)[i] - B[i].dot(A_last_error_list[i]))**2) for i in range(self.k)]
        error_star = sum(error_star)
        return error, error_star
        
    
    def F(self, X_sum, B, A):
        '''
        input is lists of the elements
        output list of elements
        '''
        # 4a  
        B = np.asarray(B)
        A = np.asarray(A)
        
        coder = SparseCoder(dictionary=B.T, transform_alpha=self.reg_lambda, transform_algorithm='lasso_cd')    
        # B: basis function 
        # A: activation function   
        B_hat, A_hat = librosa.decompose.decompose(X_sum, transformer=coder) 
        A_hat = self._pos_constraint(A_hat)

        return A_hat

    def DD(self, B, A):
        '''
        Taking the parameters as x_train_use and discriminate over the
        entire region
        '''
        # step 3
        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        
        # step 4 
        change_B = 1 
        t = 0
        
        acc_ddsc = []
        err_ddsc = []
        a_ddsc = []
        b_ddsc = []
        
        X_sum = self.train_sum # change df to list of list   
        X_train = self.train_set
        
        while t <= self.steps and self.epsilon <= change_B:
            B_cat_p = B_cat
            
            # step 4a
            A_hat = self.F(X_sum, B_cat, A_star)
            
            # step 4b
            B_cat = (B_cat - self.alpha * ((X_sum - B_cat.dot(A_hat)).dot(A_hat.T) - (X_sum - B_cat.dot(A_star)).dot(A_star.T)))
            
            # step 4c
            B_cat = self._pos_constraint(B_cat) # scale columns s.t. b_i^(j) = 1
            B_cat /= sum(B_cat)
            
            change_B = np.linalg.norm(B_cat - B_cat_p)
            t += 1
            
    
            print("step {t}: B change = {c:.4f}".format(t=t, c=change_B))

            # convergence check
            A_hat_split = np.split(A_hat, self.k, axis=0)
            B_split = np.split(B_cat,self.k,axis=1)
            
            acc_iter = self.accuracy(X_train, X_sum, B, A_hat_split)
            acc_iter = self.accuracy(X_train, X_sum, B_split, A)
            err_iter = self.error(X_train, X_sum, B, A_hat_split)

#             error, error_star = sc.error(list(x_train.values()),train_sum,B_list,A_list)

               
            acc_ddsc.append(acc_iter)
            err_ddsc.append(err_iter)
            a_ddsc.append(np.linalg.norm(A_hat))
            b_ddsc.append(np.linalg.norm(B_cat))

        self.acc_ddsc = acc_ddsc
        self.err_ddsc = err_ddsc
        self.a_ddsc = a_ddsc
        self.b_ddsc = b_ddsc
        return B_cat

    def predict(self, A, B):
        result = [x.dot(y) for (x, y) in zip(B, A)]
        return result 
    
    
if __name__ == "__main__":
    pass

