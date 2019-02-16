# coding: utf-8

# In[ ]:

#versão 2.6.1
#Inclui: possibilidade de parametrização das funções de normalização e binarização
#print horario de inicio
#incluiu criador de variáveis
#2.4.1 -- incluiu metodo de validação cruzada ordenanda e as padrões tal como unbalanced_grid_cros
#2.4.2 -- incluiu metodo de NearZeroVariance
#2.4.2 -- incluiu metodo de para variar o thershold de uma classificação afim de otimizar alguma metrica sugerida
#2.5.0 -- Inclui balanceamento por make_imbalance
#2.5.0 -- Documenta todos os métodos
#2.5.1 -- Adiciona possibilidade de otimizar qualquer parâmetro para multiclass não só mais a acurácia ('micro','macro','weigthed')
#2.5.1 -- Aceita que passe uma matrix ao invés de um dataframe
#2.6.0 -- Classe para TextMining Incluida
#2.6.1 -- Flag para ativar o alerta ou desativar
#2.6.1 -- Correcao nas metricas para regressao
#2.6.1 -- Adicionado r2_score como possivel metrica de otimizacao
#2.6.1 -- Adicionado SVR para os modelos internos de regressao



class modelagem():
    
    
    def __init__(self,tp_modelo):
        
        """
        Classe para otimizar e encapsular metodologias repetitivas de modelagem de dados
            -tp_modelo: 'class' para classificação, 'regr' para regressão e 'clus' para clusterização.
        """
        
        import warnings
        warnings.filterwarnings("ignore")
        self.tp_grid = "regular"
        self.cv = 3
        self.tp_modelo = tp_modelo
    
    def alerta_final(self):
        if self.alerta ==1:
            from IPython.display import display,Javascript
            return(display(Javascript("""
            require(
                ["base/js/dialog"], 
                function(dialog) {
                    dialog.modal({
                        title: 'Alerta!',
                        body: 'Atenção, sua execução finalizou.',
                        buttons: {
                            'OK': {}
                        }
                    });
                }
            );
            """)))
        
    def model_experiment(self,params,x_treino,x_teste,y_treino,y_teste,tp_grid,tp_otimo, categ = [],possiveis_var = [],modelo= "",
                         lgb_param = {}, pop_siz = 50, gen_num = 10, cv = 3,combinacao = 0, num_var = 10, pol =2, log =0,
                         rand =0, m_class_average = 'binary', verbose = 0, alerta = 0, grid_jobs= -1):
        
        """
        Principal método para realizar Cross Validation e GridSearch
            -modelo: método de modelagem 
                class -- >(gbm, tree, rf, knn, lgb, ada, logit, elastic, svc, gausNB, multiNB, bernNB)
                regr -- >(gbm, tree, rf, knn, linRegr)
            -params: dicionario com os parametrôs que deseja testar o modelo (se vazio usar tp_grid = 'none')
            -x_treino: dataframe de features de treino
            -y_treino: dataframe da resposta do treino
            -x_teste: dataframe de features de teste
            -y_teste: dataframe da resposta do teste
            -tp_grid: tipo de grid search: regular (para força bruta); evol (para busca genetica), random (busca aleatoria) e none para não usar grid search
            -tp_otimo: métrica a ser otimizada: recall/precision/acc/f1/mse/mae
            -categ: variáveis categóricas para realizar combinação de variáveis
            -possiveis_var: lista de possíveis variáveis a realizar combinações
            -lgb_param: dicionario de parametros para método lgb (lightGradientBoosting)
            -pop_size: tamanho da amostra para realizar otimização genética
            -gen_num: número de geração do algorítmo genético
            -cv: tipo de validação cruzada, se passar um inteiro, realiza kfold com k igual a número passado, e pode passar objeto de validação cruzada do sklearn
            -combinação: flag de ativação do método para realizar combinações (1 ativo/ 0 desativa)
            -num_var: número de variáveis a realizar combinações aleatórias
            -pol: grau de polinomio que deseja elevar as num_var que serão combinadas
            -log: ativa ou desativa operação de log nas num_var escolhidas (1 ativo/ 0 desativa)
            -rand: ativa ou desativa a aleatoriedade quando passado possiveis_var ao invés de usar algo aleatório usa as variáveis passadas na lista (1 ativo/ 0 desativa)
            
            Retorna:
            -Uma tabela com diversas métricas de avaliação do modelo tanto para teste quanto para treino (métricas variam com tp_modelo)
            -Caso tenha, retorna um dataframe com as importancias das variáveis ou com os coeficientes de uma regressão
            -Para classificação retorna a matriz de confusão do teste e do treino
            -Retorna o modelo treinado com os melhores parâmetros do GridSearch
        """
        
        import time
        import pandas as pd
        start_time = time.time()          
        if combinacao == 1:
            vari_train,new_df_train = self.combinacao_variaveis(x_treino,rand =rand ,num_var = num_var, pol = pol,
                                                                log = log, categ = categ, possiveis_var = possiveis_var)
            
            vari_test,new_df_test = self.combinacao_variaveis(x_train = x_teste,num_var = num_var,rand = 1 ,
                                                         possiveis_var=list(vari_train), pol = pol, log = log, categ = categ)
            
            self.x_treino = pd.concat([x_treino,new_df_train], axis = 1)
            self.x_teste = pd.concat([x_teste,new_df_test], axis = 1)
        else:
            self.x_treino = x_treino
            self.x_teste = x_teste
        self.alerta = alerta
        self.modelo = modelo
        self.grid_jobs = grid_jobs
        self.params = params
        self.verbose = verbose
        self.y_treino = y_treino
        self.y_teste = y_teste
        self.cv = cv
        self.tp_grid = tp_grid
        self.pop_siz = pop_siz
        self.gen_num = gen_num
        self.lgb_param = lgb_param
        self.tp_otimo = tp_otimo
        print("Criando objeto do algoritmo escolhido ...")
        self.__tipo_algoritmo(modelo)
        print("Iniciando treinamento ...")
        self.__grid_search_method(m_class_average)
        print("Avalidando modelo ...")
        self.__avaliacao()
        self.__metricas()
        #data,feat_imp,cm_teste,cm_train,clf
        print("Tempo de execução: %s seg " % (time.time() - start_time))
        self.alerta_final()
        if self.tp_modelo == 'class':
            return(self.data,self.feat_imp,self.cm_teste,self.cm_train,self.fitter)
        elif self.tp_modelo == 'regr':
            return(self.data,self.feat_imp,self.fitter)
        elif self.tp_modelo == 'clus':
            return()
        else:
            return("Tipo de modelagem deve ser 'class' para classificacao, 'regr' para regressao e 'clus para clusterizacao'")
        
    def __grid_search_method(self,m_class_average):
        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import accuracy_score,make_scorer,roc_auc_score,precision_score,recall_score,f1_score,log_loss,cohen_kappa_score,         mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score
        
        if self.tp_otimo == "acc":
            self.tp_otimo = accuracy_score
        elif self.tp_otimo == "recall":
            self.tp_otimo = recall_score
        elif self.tp_otimo == "precision":
            self.tp_otimo = precision_score
        elif self.tp_otimo == "f1":
            self.tp_otimo = f1_score
        elif self.tp_otimo == "mae":
            self.tp_otimo = mean_absolute_error
        elif self.tp_otimo == "mse":
            self.tp_otimo = mean_squared_error
        elif self.tp_otimo == "r2":
            self.tp_otimo = r2_score
#aplica um gridsearch de forca bruta
        if self.tp_grid == "regular":
            if self.modelo in ['gaussNB']:
                grid = self.fitter
            else:
                if self.tp_modelo =='class':
                    grid = GridSearchCV(  estimator = self.fitter,
                                          param_grid = self.params,
                                          scoring = make_scorer(self.tp_otimo, average = m_class_average),
                                          cv = self.cv,
                                           n_jobs = self.grid_jobs)
                else:
                    grid = GridSearchCV(  estimator = self.fitter,
                                      param_grid = self.params,
                                      scoring = make_scorer(self.tp_otimo),
                                      cv = self.cv,
                                      n_jobs = self.grid_jobs)
            
                grid.fit(self.x_treino, self.y_treino)
                classif_final = grid.best_estimator_
                self.fitter = grid.best_estimator_
            self.fitter.fit(self.x_treino, self.y_treino)
#executa grid search com algoritmo evolutivo    
        elif self.tp_grid == "evol" :
            if self.tp_modelo =='class':
                grid = EvolutionaryAlgorithmSearchCV(  estimator=self.fitter,
                                                       params=self.params,
                                                       scoring=make_scorer(self.tp_otimo, average = m_class_average),
                                                       cv=self.cv,#StratifiedKFold(n_splits=2),
                                                       verbose=True,
                                                       population_size=self.pop_siz,
                                                       gene_mutation_prob=0.10,
                                                       tournament_size=3,
                                                       generations_number=self.gen_num)
                                                       #pmap = pool.map)
            else:
                grid = EvolutionaryAlgorithmSearchCV(  estimator=self.fitter,
                                                   params=self.params,
                                                   scoring=make_scorer(self.tp_otimo),
                                                   cv=self.cv,#StratifiedKFold(n_splits=2),
                                                   verbose=True,
                                                   population_size=self.pop_siz,
                                                   gene_mutation_prob=0.10,
                                                   tournament_size=3,
                                                   generations_number=self.gen_num)
                                                   #pmap = pool.map)
            grid.fit(self.x_treino, self.y_treino)
            best_model = grid.best_params_
            self.fitter.set_params(**best_model)
            self.fitter.fit(self.x_treino, self.y_treino)
#Implementa um random search         
        elif self.tp_grid == 'random':
            if self.modelo in ['logistic','gaussNB','linRegr']:
                grid = self.fitter
            else:
                if self.tp_modelo =='class':
                    grid = RandomizedSearchCV(estimator = self.fitter,
                                              param_distributions = self.params,
                                              scoring =make_scorer(self.tp_otimo, average = m_class_average),
                                              n_iter = 30,
                                              cv = self.cv,
                                              n_jobs = self.grid_jobs)
                else:
                    grid = RandomizedSearchCV(estimator = self.fitter,
                                          param_distributions = self.params,
                                          scoring =make_scorer(self.tp_otimo),
                                          n_iter = 30,
                                          cv = self.cv,
                                          n_jobs = self.grid_jobs)
            
                grid.fit(self.x_treino, self.y_treino)
                classif_final = grid.best_estimator_
                self.fitter = grid.best_estimator_
            self.fitter.fit(self.x_treino, self.y_treino)
#caso não queira otimizar parametros, apenas executar um fit
        elif self.tp_grid in ['','none']:
            if self.params != "":
                self.fitter.set_params(**self.params)
            self.fitter.fit(self.x_treino, self.y_treino)
        else:
            return (print("Tipos de otimização implementados 'regular' para gridSearchCV e evol para 'EvolutionaryAlgorithmSearchCV'."))
    
#metodo para criar o abjeto de modelagem o algoritmo escolhido
    def __tipo_algoritmo(self,modelo):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
        from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
        from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,LinearRegression
        from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
        from sklearn.naive_bayes import GaussianNB# Usado quando as features sao continuas
        from sklearn.naive_bayes import MultinomialNB# Usado quando as features sao discretas
        from sklearn.naive_bayes import BernoulliNB# Usado quando as features sao binarizadas
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.svm import SVC,SVR
        import lightgbm as lgb
        if self.tp_modelo == "class":
            if self.modelo == "gbm":
                self.fitter = GradientBoostingClassifier(random_state = 1234,verbose = self.verbose )
            elif self.modelo == "tree":
                self.fitter = DecisionTreeClassifier(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "rf":
                self.fitter = RandomForestClassifier(random_state = 1234,n_jobs= -1,verbose = self.verbose)
            elif self.modelo == "knn":
                self.fitter = KNeighborsClassifier()
            elif self.modelo == "logistic":
                self.fitter = LogisticRegressionCV(random_state = 1234,cv = self.cv ,verbose = self.verbose)
            elif self.modelo == "elastic":
                self.fitter = SGDClassifier(random_state = 1234,penalty='elasticnet',verbose = self.verbose)
            elif self.modelo == "ada":
                self.fitter = AdaBoostClassifier(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "svc":
                self.fitter = SVC(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "xgb":
                self.fitter = XGBClassifier(seed = 1234,silent = self.verbose, n_jobs=-1)   
            elif self.modelo == "gaussNB":
                self.fitter = GaussianNB(verbose = self.verbose)
                print('Aconselhado usar quando as features sao continuas')
            elif self.modelo == "multiNB":
                self.fitter = MultinomialNB(verbose = self.verbose)
                print('Aconselhado usar quando as features sao discretas')
            elif self.modelo == "bernNB":
                self.fitter = BernoulliNB(verbose = self.verbose)
                print('Aconselhado usar quando as features sao binarias')
            elif self.modelo == "lgb":#
                try:
                    tam = len(self.y_treino[self.y_treino.columns[0]].unique())
                except:
                    tam = len(np.unique(self.y_treino))
                    pass
                if tam <= 2:# se for classificação binária
                    self.fitter = mdl=lgb.LGBMClassifier(objective = 'binary',silent=True, nthread = 5,verbose = self.verbose ,
                                                         num_leaves = 64, max_bin = 512, metric = 'binary_error')
                else:#se for multiclass 
                    self.fitter = mdl=lgb.LGBMClassifier(objective = 'multiclass',num_class = tam ,silent=True, nthread = 5,
                                                        num_leaves = 64, max_bin = 512, metric = 'multi_error',verbose = self.verbose)
            else:
                self.modelo = modelo
#Regressão            
        elif self.tp_modelo == "regr":
            if self.modelo == "gbm":
                self.fitter = GradientBoostingRegressor(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "tree":
                self.fitter = DecisionTreeRegressor(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "rf":
                self.fitter = RandomForestRegressor(random_state = 1234,verbose = self.verbose)
            elif self.modelo == "knn":
                self.fitter = KNeighborsRegressor()
            elif self.modelo == "linRegr":
                self.fitter = LinearRegression()
            elif self.modelo == "svr":
                self.fitter = SVR(verbose = self.verbose)
            elif self.modelo == "xgb":
                self.fitter = XGBRegressor(seed = 1234,silent = self.verbose, n_jobs=-1)   
        return(self.fitter)
    
    def __avaliacao(self):
#predict no teino e no teste, se for classificacao faz o predict_proba tbm
        import pandas as pd
        from sklearn.metrics import accuracy_score, confusion_matrix
        self.y_pred = self.fitter.predict(self.x_treino)
        self.y_pred_test = self.fitter.predict(self.x_teste)
        if self.tp_modelo == 'class':
            if self.modelo not in ['svc','elastic']:
                self.y_proba = self.fitter.predict_proba(self.x_treino)            
                self.y_proba_test = self.fitter.predict_proba(self.x_teste)
            self.cm_train = confusion_matrix(self.y_treino,self.y_pred)
            self.cm_teste = confusion_matrix(self.y_teste,self.y_pred_test)
        return()
    
    def __metricas(self):
#verifica se é multiclass ou não
        import pandas as pd
        import numpy as np
        try:
            tam = len(self.y_treino[self.y_treino.columns[0]].unique())
        except:
            tam = len(np.unique(self.y_treino))
            pass
        y_treino = self.y_treino
        y_pred = self.y_pred
        y_pred_test = self.y_pred_test
        y_teste = self.y_teste
        lista_treino,ind = self.__calcula_metricas(tam, y_treino, y_pred)   
        lista_teste,ind = self.__calcula_metricas(tam, y_teste, y_pred_test)  
#cria tabela com métricas
        self.data = {self.modelo + '_Treino': lista_treino, self.modelo + '_Teste': lista_teste}
        self.data = pd.DataFrame(self.data)
        self.data.index = ind
        self.data = self.data[[self.modelo + '_Treino',self.modelo + '_Teste']]
        
#caso retorne importancia
        self.feat_imp = pd.DataFrame()
        if self.modelo in ['gbm','tree','rf','xgb','lgb']:
            try:
                self.feat_imp = pd.DataFrame(self.x_treino.columns)
            except:
                self.feat_imp = pd.DataFrame(pd.DataFrame(self.x_treino).columns)
            self.feat_imp['importance'] = self.fitter.feature_importances_
            self.feat_imp=self.feat_imp.sort_values('importance', ascending=False)
        elif self.modelo in ['logistic','elastic']:
            self.feat_imp = pd.DataFrame(self.fitter.coef_,columns=self.x_treino.columns)
            
    
    def __calcula_metricas(self,dominio_y, y_true, y_pred):
        from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score,log_loss,cohen_kappa_score,         mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score
        resposta = []
        ind = []
#METRICAS PARA CLASSIFICACAO BINARIA E MULTILABEL
        if self.tp_modelo == 'class':
            if dominio_y <=2:
                resposta.append(roc_auc_score(y_true,y_pred))
                resposta.append(accuracy_score(y_true,y_pred))
                resposta.append(recall_score(y_true,y_pred))
                resposta.append(precision_score(y_true,y_pred))
                resposta.append(f1_score(y_true,y_pred, pos_label=1))
                resposta.append(cohen_kappa_score(y_true,y_pred))
                resposta.append(log_loss(y_true,y_pred))
                ind = ['ROC','ACC','RECALL','PREC','F1','COHEN KAPPA','LOG LOSS']
            else:
                resposta.append(accuracy_score(y_true,y_pred ))
                resposta.append(recall_score(y_true,y_pred , average = 'weighted'))
                resposta.append(precision_score(y_true,y_pred , average = 'weighted'))
                resposta.append(f1_score(y_true,y_pred, average = 'weighted'))
                resposta.append(cohen_kappa_score(y_true,y_pred))
                #resposta.append(log_loss(y_true,y_pred))
                ind = ['ACC','RECALL','PREC','F1','COHEN KAPPA']
            return (resposta,ind)
#METRICAS PARA REGRESSAO
        elif self.tp_modelo == 'regr':
            resposta.append(mean_absolute_error(y_true,y_pred))
            resposta.append(mean_squared_error(y_true,y_pred))
#             resposta.append(mean_squared_log_error(y_true,y_pred))
            resposta.append(median_absolute_error(y_true,y_pred))
            resposta.append(r2_score(y_true,y_pred))
            ind = ['MeanAE','MSE','MeadianAE','R2']
            return (resposta,ind)
    
#modulo de balanceamento de bases para over under overUnder e over organico    
    def balance_data(balance,X,y,ratio='auto',kind='regular',n_neighbors=3,kind_sel='all', random_state = 1234):
        
        """
        Método para balanceamento de bases (classificação)
        -balance: técnica de balanceamento (smote,CnnUnder,RandOver,RandUnder,ENN,UnderOver)
        -X: dataframe das features da base a ser balanceada
        -y: dataframe da resposta da base a ser balanceada
        -ratio: proporção do balanceamento (auto,minority, majority, range de 0 a 1)
        -Kind,n_neighbors,kind_sel parâmetros específicos das técnicas ver referência http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html
        
        Retorna:
        -Dataframe das features balanceadas
        -Dataframe da resposta balanceada
        """
        
        from imblearn.over_sampling import SMOTE
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.under_sampling import CondensedNearestNeighbour
        from imblearn.under_sampling import EditedNearestNeighbours
        from imblearn.combine import SMOTEENN
        from imblearn.datasets import make_imbalance
        import pandas as pd
        if balance == "smote":
            bal = SMOTE(random_state=random_state,ratio=ratio,kind=kind)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "CnnUnder":#reduz classe majoritaria
            bal = CondensedNearestNeighbour(random_state=random_state,ratio=ratio,n_jobs=1)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "RandOver":#aumenta classe minoritaria
            bal = RandomOverSampler(random_state=random_state,ratio=ratio)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "RandUnder":#reduz classe majoritaria
            bal = RandomUnderSampler(random_state=random_state,ratio=ratio)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "ENN":#reduz classe majoritaria
            bal = EditedNearestNeighbours(random_state=random_state,ratio=ratio,n_neighbors=n_neighbors,kind_sel=kind_sel)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "UnderOver":#reduz classe majoritaria
            bal = SMOTEENN(random_state=random_state,ratio=ratio,kind=kind,n_neighbors=n_neighbors,kind_sel=kind_sel)
            X_res, y_res = bal.fit_sample(X, y)
        elif balance == "Multi":
            X_res, y_res = make_imbalance(X=X, y=y, ratio=ratio, random_state=1234)
        else:
            print("Metodo nao implementado")
        return(pd.DataFrame(X_res,columns=X.columns),pd.DataFrame(y_res,columns=y.columns))
#######################################################MODULO DATA PREP######################################################    
    def missing_treatment(base, tp_tratamento,X_train, X, valor = 0):
        
        """
        Método para trataiva de missing:
        -base: dataframde de features para correção de missing
        -tp_tratamento: tipo de tratamento dado ao missing(remove, mean, median, mode, zero, valor, model)
        -X_train: base de treino (Não se aplica a tratativa da base de teste na base de teste usa-se, por exemplo, a média obtida na base de treino)
        -valor: caso tenha-se uma valor apropriado para o missing (ex: trocar missing por "VAZIO")
        
        Retorna
        -Dataframe com as features tratadas
        """
        import pandas as pd
        import numpy as np
        base = base
        if tp_tratamento == "remove":
            if base.iloc[:][X].dtypes == "dtype('O')":
                base.loc[:][X].fillna("NaN",inplace=True)
                base = base.loc[:][base[:][X].values != "NaN"]
            else:
                base.loc[:][X].fillna(-1,inplace=True)
                base = base.loc[:][base[:][X].values != -1]

        elif tp_tratamento == "mean":
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
            imputer = imputer.fit(X_train.loc[:][[X]].values)
            base.loc[:][X] = imputer.transform(base.loc[:][[X]].values)

        elif tp_tratamento == "median":
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
            imputer = imputer.fit(X_train.loc[:][[X]].values)
            base.loc[:][X] = imputer.transform(base.loc[:][[X]].values)

        elif tp_tratamento == "mode":
            from sklearn.preprocessing import Imputer
            imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
            imputer = imputer.fit(X_train.loc[:][[X]].values)
            base.loc[:][X] = imputer.transform(base.loc[:][[X]].values)

        elif tp_tratamento == "zero":
            base.loc[:][X].fillna(0,inplace=True)

        elif tp_tratamento == "valor":
            base.loc[:][X].fillna(valor,inplace=True)

        elif tp_tratamento == "model":
            print("Ainda não implementado")

        else:
            print("Tratamento inválido!")
        return(base)
    
    def scaling(X,range_min = 0,range_max = 1):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(range_min, range_max))
        rescaledX = scaler.fit_transform(X)
        return(rescaledX)
    
    def standardize(X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        rescaledX = scaler.transform(X)
        return(rescaledX)
    
    def normalize(X):
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer().fit(X)
        normalizedX = scaler.transform(X)
        return(rescaledX)
    
    def binarize(X, threshold = 0.0):
        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold= threshold).fit(X)
        binaryX = binarizer.transform(X)
        return(rescaledX)
    
    def dummerize(X_train_final,names_to_dummies):
        
        '''
        Método para transformar variáveis categoricas em dummies
        '''
        import pandas as pd
        X_f = pd.get_dummies(X_train_final, columns=names_to_dummies)
        return(X_f)
    
    def NearZeroVariance(X_train_final,threshold):
        
        '''
        Método para se remover da base variáveis com pouca variancia (constantes)
        '''
        
        import pandas as pd
        from sklearn.feature_selection import VarianceThreshold
        colunas = X_test_final.columns
        selector = VarianceThreshold(0.05)
        X_f = selector.fit_transform(X_train_final)
        restantes = selector.get_support(indices=True)
        selecionadas = [col for col in colunas if col not in colunas[restantes]]
        X_f = pd.DataFrame(X_f,columns=colunas[restantes])
        zero_variables = selecionadas
        return(X_f,zero_variables)
    
    def arruma_var_test(x_treino,x_teste):
        
        '''
        Método para arrumar a base de teste quando se tem número diferente de variáveis entre teste e treino
        '''
        col_amais = set(x_teste.columns) - set(x_treino.columns)
        col_amenos = set(x_treino.columns) - set(x_teste.columns)
        #Adiciona o que não tem
        for i in list(col_amenos):
            x_teste[i] = 0
        #Exclui o que te a mais
        x_teste = x_teste.drop(list(col_amais),axis =1)
        x_final_test = x_teste
        return (x_final_test)
    
        
    def combinacao_variaveis(self, x_train, num_var, rand, possiveis_var = [], categ = [] , pol = 2, log = 0):
        
        '''
        Método para realizar combinação de variáveis
        -x_train: dataframe de features a se combinar
        -categ: variáveis categóricas para realizar combinação de variáveis
        -num_var: número de variáveis a realizar combinações aleatórias
        -pol: grau de polinomio que deseja elevar as num_var que serão combinadas
        -log: ativa ou desativa operação de log nas num_var escolhidas (1 ativo/ 0 desativa)
        -rand: ativa ou desativa a aleatoriedade quando passado possiveis_var ao invés de usar algo aleatório usa as variáveis passadas na lista (1 ativo/ 0 desativa)
        '''  
        import random
        import numpy as np
        if rand == 0:
            possiveis_var = []
            for i in range(0,len(x_train.columns)):
                if x_train.loc[:][x_train.columns[0]].dtypes == 'float64':
                    possiveis_var.append(i)
            new_df = x_train.iloc[:][random.sample(possiveis_var, num_var)]
            variaveis = new_df.columns
        else:
            possiveis_var2 = []
            for i in possiveis_var :
                if (x_train.loc[:][i].dtypes == 'float64'):
                    possiveis_var2.append(i)
            new_df = x_train.iloc[:][random.sample(possiveis_var2, num_var)]
            variaveis = new_df.columns

        for i in range(0,num_var-1):
            #calcula o log de cada variavel
            if log == 1:
                column1 = new_df.columns[i]
                new_df[(str(column1)+'_log')] = np.log(new_df.iloc[:][column1]).replace([np.inf,-np.inf, np.nan],0)
            #calcula polinomios    
            if pol >=2:
                column1 = new_df.columns[i]
                for w in range(1,pol+1):
                    if w >= 2:
                        new_df[(str(column1)+'_pol'+str(w))] = new_df.iloc[:][column1]**w
            #junta variáveis numéricas
            for y in range(0,num_var-1):
                if y != i :#faz as combinacoes
                    column1 = new_df.columns[i]
                    column2 = new_df.columns[y]
                    new_df[(str(column1)+'_'+str(column2))] = new_df.iloc[:][column1]*new_df.iloc[:][column2]
        #variáveis categoricas
        if categ != []:
            dict_var = {}
            for var_cat in x_train[:][categ]:
                dict_var2 = {}
                for categoria in x_train[:][var_cat].unique():
                    dict_var2[str(categoria)]= len(x_train.loc[x_train.loc[:][var_cat] == categoria][:])
                dict_var[str(var_cat)] = dict_var2
            #substitui variavel cat pela qntde q ela ocorre
            for categoria in x_train[:][categ]:#cada uma das variáveis categoricas e.g UF, COMARCA
                new_df[str(categoria) + '_QTD_TOT'] = x_train[:][categoria]#cria um df que é a nova variável categorica com q qntde do total
                dict_busca = dict_var[str(categoria)]# dicionario de busca traz só o dicionario da variável gerada e procurada
                for item in dict_busca:#para cada item dentro desse dicionario
                    new_df[str(categoria) + '_QTD_TOT'] = new_df[str(categoria) + '_QTD_TOT'].replace(item,dict_busca[item])
        if rand == 0:
            print("Variáveis escolhidas para criação de novas:")
            print(variaveis)
            
        new_df.replace(np.nan,0)
        return (variaveis,new_df)
    
    #Metodo para executar uma validacao cruzada com kfold
    def cros_val_model(model,tp_modelo, X, Y, test_ratio = 0.2, set_param = [], n_folds = 10, tp_cros_val = 'hold_out'):
        
        '''
        Método para treinar modelos com validação cruzada
        -model: técnica de modelagem a se usar
        -tp_modelo: class: classificação; regr: regressão
        -X: dataframe full de features
        -Y: dataframe full da variável resposta
        -test_ratio: proporção da sua base de teste em cado de 'hold_out'
        -set_param: dicionário de parâmetros para seu modelo (caso já não tenha passado seu modelo já com os devidos parâmetros)
        -n_folds: número de folds para 'k_fold' e para 'order_kfold'
        -tp_cross_val: técnica de validação cruzada
            -'hold_out': separa em treino e teste de acordo com test_ratio
            -'k_fold': realiza técnica de k folds de validação
            -'order_kfold': semelhante a técnica 'k_fold' porem divide a base em n_folds e usa os folds de teste de forma ordenada (EX: se n_fold = 3 então as 33,33 primeiras linhas são usadas para validação  e assim por diante)
            -'loocv': leave one out cross validantion
        Retorn:
        -Dataframe com as medidas de avaliação de acordo com o tipo de modelo (já ajustado para problemas multiclass)
        
        '''
        
        import numpy as np
        import pandas as pd
    #verifica o metodo de validacao cruzada
        met1,met2,met3,met4,met5,met6 = ([] for i in range(6))
    #se for hold out treina e ja aplica o modelo
        if tp_cros_val == 'hold_out':
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=1234)
            if set_param != []:
                model.set_params(**set_param)
            model.fit(X,Y)
            train_pred = model.predict(X_train.iloc)
            #predito no teste
            test_pred = model.predict(X_test.iloc)
            me1,me2,me3,me4,me5,me6 = metricas_col_val(train_pred,test_pred,y_train,train_indices,
                                                                 test_indices,tp_modelo = tp_modelo,
                                                                 n_class = len(Y[Y.columns[0]].unique()))
            met1.append(me1)
            met2.append(me2)
            met3.append(me3)
            met4.append(me4)
            met5.append(me5)
            met6.append(me6) 

    #kfold ordenador de acordo com a quantidade de folds    
        if tp_cros_val == 'order_kfold':
            n_lin = round(len(X)/n_folds)
            fold = []
            for i in range(0,len(X)):
                if i%n_lin == 0:
                    fold.append(i+1)
            fold.append(len(X))

            for intervalo in range(0,len(fold)):
                if (intervalo+1) < len(fold):
                    if intervalo == (len(fold)-2):
                        teste = list(range(fold[intervalo]-1,fold[intervalo+1]-1))
                    else:
                        teste = list(range(fold[intervalo]-1,fold[intervalo+1]-1))
                    tt = list(range(fold[0],fold[len(fold)-1]))
                    train = []
                    for i in tt:
                        if i not in teste:
                            train.append(i)
                    model.fit(X.loc[train][:],Y.loc[train][:])
                    y_pred_train = model.predict(X.loc[train][:])
                    y_pred_test = model.predict(X.loc[teste][:])
                    me1,me2,me3,me4,me5,me6 = metricas_col_val(y_pred_train,y_pred_test,Y,train,
                                                                         teste,tp_modelo = tp_modelo,
                                                                         n_class = len(Y[Y.columns[0]].unique()))

                    met1.append(me1)
                    met2.append(me2)
                    met3.append(me3)
                    met4.append(me4)
                    met5.append(me5)
                    met6.append(me6)



    #se for loocv ou kfold normal 
        if tp_cros_val == 'loocv' or tp_cros_val == 'kfold':
            if tp_cros_val == 'loocv':
                from sklearn.model_selection import LeaveOneOut
                cros_Val = LeaveOneOut()
            else:
                from sklearn.model_selection import KFold
                cros_Val = KFold(n_splits = n_folds)

            for train_indices, test_indices in cros_Val.split(X):
                #treina o modelo na primeira base
                model.fit(X.iloc[train_indices], Y.iloc[train_indices])
                #predito do treino
                train_pred = model.predict(X.iloc[train_indices])
                #predito no teste
                test_pred = model.predict(X.iloc[test_indices])
                me1,me2,me3,me4,me5,me6 = metricas_col_val(train_pred,test_pred,Y,train_indices,
                                                                 test_indices,tp_modelo = tp_modelo,
                                                                 n_class = len(Y[Y.columns[0]].unique()))
                met1.append(me1)
                met2.append(me2)
                met3.append(me3)
                met4.append(me4)
                met5.append(me5)
                met6.append(me6) 
        self.alerta_final()
        return(met1,met2,met3,met4,met5,met6)


    def metricas_col_val(train_pred,test_pred,y_treino,tp_modelo,n_class,train_indices = [],test_indices = []):
        
        '''
        Método usado para exibir tabela de métricas de avaliação (usado internamente)
        '''
        
        from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score,log_loss,cohen_kappa_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score
        acc_test = acc_train = rec_train = rec_test = pre_train = prec_test = []
        mae_train = mae_test = mse_train = mse_test = rs2_train = rs2_test = []
        if tp_modelo == 'class':
            if n_class < 2: #caso seja classificacao binaria
                acc_train=accuracy_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices])
                acc_test=accuracy_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices])
                rec_train=recall_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices])
                rec_test=recall_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices])
                pre_train=precision_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices])
                prec_test=precision_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices])
                return(acc_train,acc_test,rec_train,rec_test,pre_train,prec_test)
            else:#caso seja multiclass
                acc_train=(accuracy_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices]))
                acc_test=(accuracy_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices]))
                rec_train=(recall_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices],average = 'weighted'))
                rec_test=(recall_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices],average = 'weighted'))
                pre_train=(precision_score(y_pred=train_pred,y_true=y_treino.iloc[train_indices],average = 'weighted'))
                prec_test=(precision_score(y_pred=test_pred,y_true=y_treino.iloc[test_indices],average = 'weighted'))
                return(acc_train,acc_test,rec_train,rec_test,pre_train,prec_test)
        if tp_modelo == 'regr': # em caso de regressao
            mae_train=(mean_absolute_error(y_pred=train_pred,y_true=self.y_treino.iloc[train_indices]))
            mae_test=(mean_absolute_error(y_pred=test_pred,y_true=self.y_treino.iloc[test_indices]))
            mse_train=(mean_squared_error(y_pred=train_pred,y_true=self.y_treino.iloc[train_indices]))
            mse_test=(mean_squared_error(y_pred=test_pred,y_true=self.y_treino.iloc[test_indices]))
            rs2_train=(r2_score(y_pred=train_pred,y_true=self.y_treino.iloc[train_indices]))
            rs2_test=(r2_score(y_pred=test_pred,y_true=self.y_treino.iloc[test_indices])) 
            return(mae_train,mae_test,mse_train,mse_test,rs2_train,rs2_test)
        
        
    def total_possibilidade(self,params):
        
        '''
        Método criado para retornar lista com todas as possibilidades de um dicionário para realizar GridSearch na mão
        '''
        
        param_final = []
        tot_poss = 1
        for i in params.keys():
            tot_poss = tot_poss * (len(params[str(i)]))

        lista_de_dict = [{} for poss in range(0,tot_poss)]

        n_keys_aux = -1
        for key in params.keys():#para cada keys do dicionario principal
            n_keys = len(params[str(key)])# qtd de elementos dentro da key
            for itens in range(0,n_keys): #para cada elemento dentro da key
                if n_keys_aux ==-1: #primeira iteração escreve tudo
                    for ix in range(int(itens*(tot_poss/n_keys)),int((itens+1)*(tot_poss/n_keys))):
                        lista_de_dict[ix][key] = params[str(key)][itens]

                else: # dai rpa frente
                    passo = tot_poss/n_keys_aux #passo = 10 
                    aux = tot_poss/n_keys # aux = 6 intervalo de escrita
                    aux2 = aux/n_keys_aux # aux2 = 2 intervalo de escrita anterior

                    for ix in range(0,tot_poss):
                        if ix%passo < aux2: # se for menor que o intervalo de escrita
                            lista_de_dict[ix+int(aux2*itens)][key] = params[str(key)][itens]          

            n_keys_aux = n_keys
        return(tot_poss,lista_de_dict)

    def cross_grid_unbalanced(self,model,balanced_X,balanced_Y, unbalanced_X, unbalanced_Y,params,met_otm, n_folds = 5, ratio_split = 0.9):
        
        '''
        Método de validação cruzada para problemas de variável resposta desbalanceada, onde otimiza os parâmetros para uma base de validação desbalanceada
        -model: técnica de modelagem que se deseja aplicar
        -balanced_X: dataframe de features balanceada
        -balanced_Y: dataframe de resposta balanceada
        -unbalanced_X: dataframe de features desbalanceadas
        -unbalanced_Y: dataframe de resposta desbalanceada
        -params: dicionario de parâmetros a serem otimizados
        -met_otm: objeto do sklearn.metrics para seleção de métrica a ser otimizada
        -n_folds: número de folds para realizar validação cruzada (aceita apenas k fold)
        -ratio_split: proporção da base que se deseja usar para fazer uma amostragem dos folds
        
        Retorna
        -Dataframe com todas as possibilidades testadas e com todas as métricas obtidas no treino e na validação desbalanceada
        '''
        
        import time
        import random
        import pandas as pd
        start_time = time.time() 
        metrica_train = []
        metrica_test = []
        grid_params = []
        tables_final = pd.DataFrame()
        possibilidades,lista_dict = self.total_possibilidade(params)
        for fold in range(1,n_folds):
            #cria amostra aleatoria da base balanceada com 90% da base
            random_train = random.sample(list(balanced_X.index),round(len(balanced_X)*ratio_split))
            random_teste = random.sample(list(unbalanced_X.index),round(len(unbalanced_X)*ratio_split))
            for i in range(0,possibilidades):
                #fold de treino
                xtreino = balanced_X.loc[random_train][:]
                ytreino = balanced_Y.loc[random_train][:]
                #fold de teste
                xteste = unbalanced_X.loc[random_teste][:]
                yteste = unbalanced_Y.loc[random_teste][:]
                #treina o modelo
                model.set_params(**lista_dict[i])
                model.fit(xtreino,ytreino)
                #predict no modelo
                y_pred_train = model.predict(xtreino)
                y_pred_test = model.predict(xteste)
                #metricas de avaliação
                metrica_train.append(met_otm(y_pred=y_pred_train,y_true=ytreino))
                metrica_test.append(met_otm(y_pred=y_pred_test,y_true=yteste))
                grid_params.append(lista_dict[i])

        tables_final['Parametros'] = grid_params
        tables_final['Treino'] = metrica_train
        tables_final['Teste'] = metrica_test
        print("Tempo de execução: %s seg " % (time.time() - start_time))
        self.alerta_final()
        return(tables_final)
    
    def best_thershold(model,X_teste,y_true, qtd_perc = 100, otm = 'F1'):
        
        """
        Método para testar qual melhor thershold para problemas de classificação:
        -model: técnica de modelagem escolhida
        -X_teste: dataframe de teste para verificar o corte
        -y_true: dataframe das respostas da base de features de teste  acima
        -qtd_perc: divisão das faixas 1/qtd_perc
        -otm: parâmetro de otimização (aceita, f1,prec,rec)
        
        Retorna:
        -Dataframe com todas as faixas escolhidas, as métricas testadas, a matriz de confusão 
        -Dataframe com o melhor corte e suas respectivas métricas
        """
        
        import pandas as pd
        from sklearn.metrics import recall_score,confusion_matrix,precision_score,f1_score
        prob = model.predict_proba(X_teste)
        thershold,prec,rec,f1,cm = ([] for i in range(5))
        final_df = pd.DataFrame()
        for i in range(1,qtd_perc):
            prob_final = []
            for y in range(0,len(prob)):
                if prob[y][1] >= i/qtd_perc:
                    prob_final.append(1)
                else:
                    prob_final.append(0)
            thershold.append(i/qtd_perc)
            prec.append(precision_score(y_pred=prob_final, y_true=y_true))
            rec.append(recall_score(y_pred=prob_final, y_true=y_true))
            f1.append(f1_score(y_pred=prob_final, y_true=y_true))
            cm.append(confusion_matrix(y_pred=prob_final, y_true=y_true))

        final_df['thershold'] = thershold
        final_df['Precision'] = prec
        final_df['Recall'] = rec
        final_df['F1'] = f1
        final_df['Confusion Matrix'] = cm
        best = final_df[final_df[otm] == final_df.loc[:][otm].max()]
        return(final_df,best)
    
    
class text_mining_full(modelagem):
    """
    Classe que herda a classe de modelagem só que com um vies para TextMining, todas as funções da outra classe se encontram
    aqui quando validas para classificação de TextMining
    """
    
    def __init__(self, tp_modelo):
        modelagem.__init__(self,tp_modelo)
        
    def prep_text(self,text,stp_lang = "portuguese",stp_wr = 1, stemm = 1, lista_Xclusive = [], caract_spec = 1, double_space =1):
        """
        Função responsável por realizar a preparação dos dados para dados do tipo texto
        Parâmetros:
        text: string, ou lista de strings ou coluna de pandas DataFrame
        stp_lang: Idioma para realizar o stop words, default = "portuguese"
        stp_wr: 1 ativa exclusão de stop words, diferente de 1 não permite
        stemm: 1 ativa truncamento por stemm, diferente de 1 não permite
        stemm: se lista não vazia exclue palavras passadas na lista senão não
        caract_spec: se 1 mantém apenas letras sem acentuação, se 2 mantém letra e número, se 3 limpa apenas número e se 4 retira apenas caracteres não referente a lingua mas a expressões
        double_space: se 1 substitui espaços duplos por simples 
        """
                
        from nltk.tokenize import sent_tokenize, word_tokenize     
        new_text = []
        for i in text:
            if len(text) > 1:
                interm_text = i
            else:
                interm_text = text[0]

            #remove lista de palavras passadas por parametro
            if lista_Xclusive != []:
                interm_text = ' '.join([word for word in word_tokenize(interm_text) if word not in lista_Xclusive])
            #limpa stop_words
            if stp_wr == 1:
                from nltk.corpus import stopwords
                interm_text = ' '.join([word for word in word_tokenize(interm_text) if word not in stopwords.words(stp_lang)])
            #realiza truncamento por stemm
            if stemm == 1:
                from nltk.stem import RSLPStemmer
                interm_text = ' '.join([RSLPStemmer().stem(word) for word in word_tokenize(interm_text)])
            #Limpa Caracter que não seja letra
            if caract_spec == 1:
                import re
                interm_text = re.sub('[^A-Za-z]+', ' ', interm_text)
            #limpa caracter que não seja letra nem número
            if caract_spec == 2:
                import re
                interm_text = re.sub('[^A-Za-z0-9]+', ' ', interm_text)
            #limpa caracter que seja número
            if caract_spec == 3:
                import re
                interm_text = re.sub('[0-9]+', ' ', interm_text)
            #limpa caracter que seja número
            if caract_spec == 4:
                import re
                import string
                regex = re.compile('[%s]' % re.escape(string.punctuation))
                interm_text = regex.sub('',interm_text)
            #remove espaço duplo
            if double_space ==1:
                import re
                interm_text = re.sub('\W+',' ', interm_text )

            new_text.append(interm_text)
        return(new_text)
    
#     def text_data_format(self):
        
        
    def textos_organicos(self,txt,model,ratio_txt=0.3, n_samples = 2):
        """
        Método para criação de textos organicos a partir de embedding
        Recebe:
        -Coluna do dataframe com cada linha contendo um texto
        -model: modelo responsável por realizar o embedding
        -ratio_txt: percentual de alteração do texto original
        -n_samples: quantidade de geração de novos textos
        Retorna:
        -Lista contendo uma novo textos por elemento da lista
        """
        import random
        lista_f = []
        for i in range(0,len(txt)):
            lista_f.append(txt[i].lower().split())

        lista_sor = []
        lista_novos = []
        for y in range(0,n_samples):
            txt = random.sample(lista_f,1)[0]#pega um dos textos aleatorios
            txt_novo = txt[:]
            txt_rnd = random.sample(range(0,len(txt_novo)),round(len(txt_novo)*ratio_txt))#seleciona as palavras aleatorias dentro desse txt

            for ind in txt_rnd:
                try:
                    palavras_sin = random.sample(model.most_similar(txt_novo[ind],topn=3),1)[0][0]#acha uma palavra sinonimo para cada palavra do ratio
                    txt_novo[ind] = palavras_sin
                except:
                    print("Palavra não contida no dicionário: ",txt_novo[ind])
                    pass
            lista_novos.append(txt_novo)
        lista_f = lista_f + lista_novos
        for i in lista_f:
            lista_sor.append(" ".join(i))
        return(lista_sor)
    
    def embeddings(self,X,y, tp_representation = "bow", split_ratio = 0.7, random_state = 1234,
                    min_df=1, max_df = 1, n_gram= 1, model = "",w2idf = None, pd = True):

        """
        Método para realizar embedding com 3 implementações internas: frequencia, tf/idf e word2vec por média das palavras na frase

        Parâmetros:
        X: variável explicativa no formato de texto para realizar o embedding
        y: variável resposta
        tp_representation: tipo de embedding ("bow": para bag of words simple (default); "tfidf": para term frequency/ inverse document
        frequency; "word2vec" para algoritmo word2vec ou glove ou similar.
        split_ratio: distribuição da base de treino e teste default = 0.7
        random_state: default = 1234
        min_df: default = 1
        max_df:default = 1
        n_gram: default =  1
        model: modelo a ser passado para o caso de word2vec/glove/wang2vec etc.

        Retorna:
        -Se for 'bow' ou 'tfidf'
        X_train: dataframe transformado pelo embedding escolhido
        X_test: dataframe transformado pelo embedding escolhido com aprendizado do treino
        y_train: dataframde da resposta de treino
        y_test: dataframde da resposta de treino
        emb: método do embedding para aplicar em novas bases
        
        -Se for 'word2vec':
        dataframe transformado de acordo com o modelo passado com as colunas nomeadas sequencialmente por peso do embedding
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=1234)

        if tp_representation == "bow":
            emb = CountVectorizer(min_df=min_df, max_df = max_df, ngram_range=(1,n_gram))            
            X_train = emb.fit_transform(X_train)
            X_test = emb.transform(X_test)
        elif tp_representation == "tfidf":
            emb = CountVectorizer(min_df=min_df, max_df = max_df, ngram_range=(1,n_gram))            
            X_train = emb.fit_transform(X_train)
            X_test = emb.transform(X_test)
        elif tp_representation == 'word2vec':
            coluna_final = [] #vetor inicial 
            for frase in X:
                cont = 0
                soma = np.repeat(0., model.vector_size)# cria vetor do tamanho do vetor de embedding passado do modelo
                frase = frase.replace("{", "").replace("}", "").lower()#data prep na frase
                for word in frase.split(" "):#para cada palavra dentro da frase
                    try:
                        if w2idf is not None:#se nenhum w2idf é passado 
                            soma = np.add(soma, np.multiply(model.get_vector(word), w2idf[word]))#realiza soma de cada palavra 
                            cont += w2idf[word]#calcula o denominador para média 
                        else:
                            soma += np.add(soma, model.get_vector(word))
                            cont += 1
                    except KeyError as ke:
                        pass
                coluna_final.append(np.divide(soma, cont))
        if tp_representation != 'word2vec':
            return(X_train, X_test, y_train, y_test, emb)
        else:
            return(pd.DataFrame(coluna_final,columns=['emb_'+ str(i) for i in range(1,len(coluna_final[0])+1)]))

    def wordcloud_draw(data, color = 'black'):
        from wordcloud import WordCloud,STOPWORDS
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        words = ' '.join(data)
        wordcloud = WordCloud(
                          background_color=color,
                          width=2500,
                          height=2000
                         ).generate(words)

        plt.figure(1,figsize=(13, 13))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
        
class exPlotRatory():
    """
    Classe para explorar os dados de uma forma grafica com simplificacoes de analises usuais em ciencia de dados
    """
    
    def scatter(self):
        """
        -Monta Scatter plot das variaveis explicativas
        """
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        
        pd.scatter_matrix(X, figsize=(8, 8))
        plt.show()
    
    def plot_per_target(X,y):
        """
        -Monta grafico para cada classe da resposta versus cada variavel explicativa passada
        """
        if str(type(X)) != "<class 'pandas.core.frame.DataFrame'>":
            X = pd.DataFrame(X,columns=range(1,len(X)))

        if str(type(y)) != "<class 'pandas.core.frame.DataFrame'>":
            y = pd.DataFrame(y,columns=[['target']])

        num_class = np.unique(y)

        dataframe = pd.concat([X,y],axis=1)

        x = ""

        for features in range(0,len(X.columns)):
    #     for classes in num_class:
            if X.iloc[features].dtypes == 'float64':                
                plt.hist(dataframe[:][X.columns[features]])
                plt.title('Variavel explicativa ' + str(X.columns[features]))
                plt.show()
                x = input("Para cancelar pressione qualquer tecla.")
                if x != "":
                    return()   

    def plot_per_target2(X,y,var,colors_x = []):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rc
        import pandas as pd
        #verifica se é um dataframe, caso não seja ela converte
        if str(type(X)) != "<class 'pandas.core.frame.DataFrame'>":
            X = pd.DataFrame(X,columns=range(1,len(X)))
        #verifica se é um dataframe, caso não seja ela converte
        if str(type(y)) != "<class 'pandas.core.frame.DataFrame'>":
            y = pd.DataFrame(y,columns=[['target']])
        else:
            y.columns = ['target']
        #cria vetor unico das classes
        num_class = np.unique(y)

        #cria uma dataframe que vai servir para criar o df final de plotagem
        dataframe = pd.concat([X,y],axis=1)
        plt.figure(figsize=(8,8))
        if dataframe[var].dtype == "object" or dataframe[var].dtype == "int32":
            #cria esqueleto para dataframe de plotagem
            ll = pd.DataFrame(index=np.unique(X[var]),columns=np.unique(y))
            ll = ll.fillna(float(0))
            #de acordo com o que for passado pelo usuario ele cria uma tabela especifica para plotar o grafico de barrar empilhadas
            for colunas in ll.columns:
                for linhas in ll.index:
                    total = len(dataframe[dataframe[var]==linhas])
                    try:
                        valor = len(dataframe[(dataframe.target == colunas) & (dataframe[var] == str(linhas))])/total
                    except:
                        valor = len(dataframe[(dataframe.target == colunas) & (dataframe[var] == linhas)])/total
                    ll.set_value(index=linhas,col=colunas,value=float(valor))

            # Data
            r = ll.index

            # Gráfico
            #espessura das barras
            acum = 0
            #dinamicamente vai empilhando as barras
            patches = []
            for i in num_class:
                if colors_x == []:#caso queiram-se as cores padrao
                    patches.append(plt.bar(r, ll[i],bottom= acum, width=0.7,label = i,edgecolor=['black']*len(r)))
                else:#caso passe um vetor de cores
                    patches.append(plt.bar(r, ll[i],bottom= acum,color =colors_x[i] , width=0.7,label = i,edgecolor=['black']*len(r)))
                acum = acum + ll[i]
            plt.bar.edgecolor = 'black'
            #adiciona text intermediário nas barras
            for j in range(len(patches)):
                for i, patch in enumerate(patches[j].get_children()):
                    bl = patch.get_xy()
                    x = 0.5*patch.get_width() + bl[0]
                    y = 0.5*patch.get_height() + bl[1]
                    plt.text(x,y, "%.2f" % (ll[j][i]), ha='center', fontsize = 15)

            # Títulos dos eixos
            plt.margins(y=0.2)
            plt.xticks(r, ll.index, fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlabel("Variável Explicativa :" + var, fontsize = 12)
            plt.legend(loc = 0 ,ncol = len(num_class),fancybox = True, fontsize = 12)
            # Plota gráfico
            plt.show()

        else:
            plt.figure(figsize=(8,8));
            for i in num_class:
                if colors_x == []:
                    plt.hist(dataframe[dataframe.target == i][var],alpha = 0.5,edgecolor='black');
                else:
                    plt.hist(dataframe[dataframe.target == i][var],alpha = 0.5,edgecolor='black',color = colors_x[i]);
            plt.margins(y=0.2);
            plt.xlabel("Variável Explicativa :" + var,fontsize = 12);
            plt.yticks(fontsize = 15)
            plt.xticks(fontsize = 15)
            plt.legend(num_class, loc = 0 ,ncol = len(num_class),fontsize = 12);
            plt.show()
        
