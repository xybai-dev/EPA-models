import os
import tensorflow as tf

def get_inputs():
    input_pos=tf.placeholder(tf.float32,[None,601],name="X_pos")
    input_noise=tf.placeholder(tf.float32,[None,3],name="X_neg")
    input_neg=tf.placeholder(tf.float32,[None,601],name="latent")
    input_latent=tf.placeholder(tf.float32,[None,3],name="latent")
    return input_pos,input_noise,input_neg,input_latent
    
def generate_cv4_data(df_new):
    allele_ls= list(set(df_new["seq"]))
    pep_ls= list(set(df_new["smiles1"]))
    while True:
        for ii in range(200,400):
            for jj in range(2900,4500):
                exstract_allele=random.sample(allele_ls, ii)
                exstract_pep=random.sample(pep_ls, jj)
                df_train = df_new[df_new["seq"].isin(exstract_allele) & df_new["smiles1"].isin(exstract_pep)]
                df_test = df_new[~(df_new["seq"].isin(exstract_allele) | df_new["smiles1"].isin(exstract_pep))]
                pos_neg_ratio=(len(df_train[df_train["log_aff"]>=6.3])+len(df_test[df_test["log_aff"]>=6.3]))/(len(df_train[df_train["log_aff"]<6.3])+len(df_test[df_test["log_aff"]<6.3]))
                if (len(df_train)/len(df_test)>=3.9 and len(df_train)/len(df_test)<=4.1) and (len(df_train[df_train["log_aff"]>=6.3])/len(df_train[df_train["log_aff"]<6.3])/pos_neg_ratio<=1.1) and (len(df_train[df_train["log_aff"]>=6.3])/len(df_train[df_train["log_aff"]<6.3])/pos_neg_ratio>=0.9):
                    break
            break
        break
#     print("CV4", len(df_train),len(df_test),len(df_train)/len(df_test))
    X_train= np.array([list(np.concatenate((df_train["mol2vec"].values[i].vec, df_train["kinase_vec"].values[i]), axis=0)) for i in range(len(df_train))])
    y_train = np.array([float(ii) for ii in  df_train['log_aff'].values])
    X_test= np.array([list(np.concatenate((df_test["mol2vec"].values[i].vec, df_test["kinase_vec"].values[i]), axis=0)) for i in range(len(df_test))])
    y_test = np.array([float(ii) for ii in  df_test['log_aff'].values])
    return X_train, y_train, X_test, y_test
def encoder(input_pos,reuse=False,alpha=0.01):
    with tf.variable_scope("encoder",reuse=reuse):
        layer1=tf.layers.dense(input_pos,512)
        layer1=tf.maximum(alpha*layer1,layer1)
#         layer1=tf.nn.dropout(layer1,keep_prob=0.75)

        layer2=tf.layers.dense(layer1,256)
        layer2=tf.layers.batch_normalization(layer2, training=True)
        layer2=tf.maximum(alpha*layer2,layer2)
#         layer2=tf.nn.dropout(layer2,keep_prob=0.75)

        layer3=tf.layers.dense(layer2,128)
        layer3=tf.layers.batch_normalization(layer3, training=True)
        layer3=tf.maximum(alpha*layer3,layer3)
#         layer3=tf.nn.dropout(layer3,keep_prob=0.75)

        layer4=tf.layers.dense(layer3,64)
        layer4=tf.layers.batch_normalization(layer4, training=True)
        layer4=tf.maximum(alpha*layer4,layer4)
#         layer4=tf.nn.dropout(layer4,keep_prob=0.75)

        latent=tf.layers.dense(layer4,3)

        return latent


def decoder(input_latent,is_train=True,alpha=0.01):
    with tf.variable_scope("decoder",reuse=(not is_train)):
        hidden1=tf.layers.dense(input_latent,64)
#         hidden1=tf.layers.batch_normalization(hidden1, training=is_train)
        hidden1=tf.maximum(alpha*hidden1,hidden1)
#         hidden1=tf.layers.dropout(hidden1,rate=0.25)

        hidden2=tf.layers.dense(hidden1,128)
#         hidden2=tf.layers.batch_normalization(hidden2, training=is_train)
        hidden2=tf.maximum(alpha*hidden2,hidden2)
#         hidden2=tf.layers.dropout(hidden2,rate=0.25)

        hidden3=tf.layers.dense(hidden2,256)
#         hidden3=tf.layers.batch_normalization(hidden3, training=is_train)
        hidden3=tf.maximum(alpha*hidden3,hidden3)
#         hidden3=tf.layers.dropout(hidden3,rate=0.25)

        hidden4=tf.layers.dense(hidden3,512)
#         hidden4=tf.layers.batch_normalization(hidden4, training=is_train)
        hidden4=tf.maximum(alpha*hidden4,hidden4)
#         hidden4=tf.layers.dropout(hidden4,rate=0.25)

        outputs=tf.layers.dense(hidden4,601)
        return outputs


def generator(input_noise,is_train=True,alpha=0.01):
    with tf.variable_scope("generator",reuse=(not is_train)):
        hidden1=tf.layers.dense(input_noise,64)
        hidden1=tf.layers.batch_normalization(hidden1, training=is_train)
        hidden1=tf.maximum(alpha*hidden1,hidden1)
#         hidden1=tf.layers.dropout(hidden1,rate=0.25)

        hidden2=tf.layers.dense(hidden1,16)
        hidden2=tf.layers.batch_normalization(hidden2, training=is_train)
        hidden2=tf.maximum(alpha*hidden2,hidden2)
#         hidden2=tf.layers.dropout(hidden2,rate=0.25)

        outputs=tf.layers.dense(hidden2,3)

        return outputs
    
def discriminator1(input_latent,reuse=False,alpha=0.01):
    with tf.variable_scope("discriminator1",reuse=reuse):
        layer1=tf.layers.dense(input_latent,512)
        layer1=tf.maximum(alpha*layer1,layer1)

        layer2=tf.layers.dense(layer1,256)
        layer2=tf.maximum(alpha*layer2,layer2)

        layer3=tf.layers.dense(layer2,128)
        layer3=tf.maximum(alpha*layer3,layer3)

        layer4=tf.layers.dense(layer3,64)
        layer4=tf.maximum(alpha*layer4,layer4)

        layer5=tf.layers.dense(layer4,3)
        layer5=tf.maximum(alpha*layer5,layer5)

        logits= tf.layers.dense(layer5,1)
        outputs=tf.sigmoid(logits)

        return logits, outputs 
    
def discriminator2(input_latent,reuse=False,alpha=0.01):
    with tf.variable_scope("discriminator2",reuse=reuse):
        layer1=tf.layers.dense(input_latent,512)
        layer1=tf.maximum(alpha*layer1,layer1)

        layer2=tf.layers.dense(layer1,256)
        layer2=tf.maximum(alpha*layer2,layer2)

        layer3=tf.layers.dense(layer2,128)
        layer3=tf.maximum(alpha*layer3,layer3)

        layer4=tf.layers.dense(layer3,64)
        layer4=tf.maximum(alpha*layer4,layer4)

        layer5=tf.layers.dense(layer4,3)
        layer5=tf.maximum(alpha*layer5,layer5)

        logits= tf.layers.dense(layer5,1)
        outputs=tf.sigmoid(logits)

        return logits, outputs 
def get_loss(input_pos,input_neg,input_noise,smooth=0.1):
    g_outputs=generator(input_noise,is_train=True)
    input_latent_pos=encoder(input_pos)
    input_latent_neg=encoder(input_neg, True)
    
    d_logits_pos,d_outputs_pos=discriminator1(input_latent_pos)
    d_logits_false1,d_outputs_false1=discriminator1(g_outputs,reuse=True)
    d1_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_pos,
                                                                      labels=tf.ones_like(d_logits_pos)*(1-smooth)))
    d1_loss_false=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_false1,
                                                                      labels=tf.zeros_like(d_logits_false1)))
    d1_loss=tf.add(d1_loss_real,d1_loss_false)
    
    d_logits_neg,d_outputs_neg=discriminator2(input_latent_neg)
    d_logits_false2,d_outputs_false2=discriminator2(g_outputs,reuse=True)
    d2_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_neg,
                                                                      labels=tf.zeros_like(d_logits_neg)))
    d2_loss_false=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_false2,
                                                                      labels=tf.ones_like(d_logits_false2)*(1-smooth)))
    d2_loss=tf.add(d2_loss_real,d2_loss_false)
    
    
    g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_false1,
                                                                      labels=tf.ones_like(d_logits_false1)*(1-smooth)))    
    
    g_loss_2=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_false2,
                                                                      labels=tf.zeros_like(d_logits_false2)))    
    
    
    g_loss=tf.add(g_loss_1,g_loss_2)
    outputs=decoder(input_latent_pos)
    ae_loss=tf.reduce_mean(tf.square(outputs-input_pos))
    return g_loss,d1_loss,d2_loss,ae_loss

def plot_image(samples,step):
    y_gen=[]
    for ii in samples:
        if 6.3<=ii[-1]<11:
            y_gen.append(ii[-1])
    if len(y_gen)!=0:
        plt.figure(figsize=(5,2))
        sns.kdeplot(y_gen,label=step)
    
def stat_y(samples):
    y_gen=[]
    for ii in samples:
        if 6.3<ii[-1]<11:
            y_gen.append(ii[-1])
    if len(y_gen)>0:
        return len(y_gen),max(y_gen)-min(y_gen)
    
    else:
        return len(y_gen),0
        
        
def get_optimizer(g_loss,d1_loss,d2_loss,ae_loss,beta1=0.4,learning_rate=0.001):
    train_vars=tf.trainable_variables()
    g_vars=[var for var in train_vars if "generator" in var.name]    
    d1_vars=[var for var in train_vars if "discriminator1" in var.name]
    d2_vars=[var for var in train_vars if "discriminator2" in var.name]
    ae_vars=[var for var in train_vars if "decoder" in var.name or "encoder" in var.name]
    g_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(g_loss,var_list=g_vars)
    d1_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d1_loss,var_list=d1_vars)
    d2_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d2_loss,var_list=d2_vars)
    ae_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(ae_loss,var_list=ae_vars)
    return g_opt, d1_opt,d2_opt,ae_opt
            
def show_generator_output(sess,n_images,input_noise,input_latent):
    noise_shape=input_noise.get_shape().as_list()[-1]
    sample_noise=np.random.normal(0,1,size=(n_images,noise_shape))
    latent=sess.run(generator(input_noise,False),feed_dict={input_noise:sample_noise})
    samples=sess.run(decoder(input_latent,False),feed_dict={input_latent:latent})
    return samples

def pcm_aae(X_all_pos,X_all_neg)
    batch_size=64
    noise_size=3

    learning_rate=0.001
    beta1=0.9
    epoch=25
    train_loss_d=0
    train_loss_c=0
    train_loss_ae=0
    nn=8
    for cc in range(20):
        tf.reset_default_graph()
        print(cc)
        checkpoint=0
        ckpt_dir = './ckpt_final/pcm_aae'+str(nn)
        nn=nn+1    
        with tf.Graph().as_default():          
            losses=[]
            step=-1
            input_pos,input_noise, input_neg,input_latent=get_inputs()
            g_loss,d1_loss,d2_loss,ae_loss=get_loss(input_pos,input_neg,input_noise)
            g_train_opt,d1_train_opt,d2_train_opt,ae_train_opt=get_optimizer(g_loss,d1_loss,d2_loss,ae_loss,beta1,learning_rate)
            with tf.Session() as sess:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                for e in range(epoch):  
                    row_rand_array = np.arange(X_all_pos.shape[0])
                    np.random.shuffle(row_rand_array)
                    row_rand = X_all_pos[row_rand_array[:len(X_all_pos)]]
                    X_pos=np.array([ii for ii in row_rand])
                    row_rand_array1 = np.arange(X_all_neg.shape[0])
                    np.random.shuffle(row_rand_array1)
                    row_rand = X_all_neg[row_rand_array[:len(X_all_pos)]]
                    X_neg=np.array([ii for ii in row_rand])
                    for batch_i in range(len(X_pos)//batch_size):
                        step=step+1
                        if batch_i !=len(X_pos)//batch_size:
                            batch_real=X_pos[batch_i*batch_size:(batch_i+1)*batch_size]
                            batch_real=batch_real.reshape(batch_size,601) 
                            batch_neg=X_neg[batch_i*batch_size:(batch_i+1)*batch_size]
                            batch_neg=batch_neg.reshape(batch_size,601) 
                        else:
                            batch_real=X_pos[batch_i*batch_size:]
                            batch_real=batch_real.reshape(batch_size,601)
                            batch_neg=X_neg[batch_i*batch_size:]
                            batch_neg=batch_neg.reshape(batch_size,601)
                   
                        
                        batch_noise=np.random.normal(0,1,size=(batch_size,noise_size))
                        _=sess.run(ae_train_opt, feed_dict={input_pos:batch_real})
                        _=sess.run(g_train_opt, feed_dict={input_noise:batch_noise})
                        _=sess.run(d1_train_opt, feed_dict={input_pos:batch_real,input_noise:batch_noise})
                        _=sess.run(d2_train_opt, feed_dict={input_neg:batch_neg,input_noise:batch_noise})

                        train_loss_d1=d1_loss.eval({input_pos:batch_real,input_noise:batch_noise})
                        train_loss_d2=d2_loss.eval({input_neg:batch_neg,input_noise:batch_noise})
                        train_loss_g=g_loss.eval({input_noise:batch_noise})
                        train_loss_ae=ae_loss.eval({input_pos:batch_real,input_neg:batch_neg})
                        losses.append((train_loss_d1,train_loss_d2,train_loss_ae,train_loss_g))
                        
                        if step==0 or step%100==0:
                            
                            print("step {}/{}".format(step,e),
                                  "Dis1 loss: {:.4f}".format(train_loss_d1),
                                  "Dis2 loss: {:.4f}".format(train_loss_d2),
                                  "AE loss: {:.4f}".format(train_loss_ae),
                                  "Generator loss: {:.4f}".format(train_loss_g))
                            samples=show_generator_output(sess,1000,input_noise,input_latent)
                            pos_len,gen_y_mean=stat_y(samples)
                            if pos_len>=400 and checkpoint<5 and gen_y_mean>=1.5: 
                                print("May save file")
                                ckpt_dir1=ckpt_dir+"_"+str(step)
                                try:
                                    X_train, y_train, X_test, y_test = generate_cv4_data(df_2new) 
                                except:
                                    X_train, y_train, X_test, y_test = generate_cv4_data(df_2new)
                                X_train1=norm_model.transform(X_train)
                                X_test1=norm_model.transform(X_test)        
                                n_samples=Counter([get_binary_class(ii) for ii in y_train])[0]-Counter([get_binary_class(ii) for ii in y_train])[1]
                                samples=show_generator_output(sess,n_samples,input_noise,input_latent)
                                df_new_sample=pd.DataFrame()
                                df_new_sample["pcm2vec"]=[ii[:-1] for ii in samples]
                                df_new_sample["aff"]=[ii[-1] for ii in samples]
                                df_new_sample=df_new_sample[[ii>=6.3 and ii<11 for ii in df_new_sample["aff"]]]
                                X_new_sample=np.array([ii for ii in df_new_sample["pcm2vec"]])
                                y_new_sample=np.array([ii for ii in df_new_sample["aff"]])
                                X_new_sample1=norm_model.transform(X_new_sample)
                                X_train_gan=np.vstack((X_train1,X_new_sample1))
                                y_train_gan=np.hstack((y_train,y_new_sample))
                                print(X_train_gan.shape)
                                mlp_norm_gan = neural_network.MLPRegressor(hidden_layer_sizes=(200,200,200),alpha=0.01, max_iter=30, solver="adam",random_state=123)
                                mlp_norm_gan.fit(X_train_gan, y_train_gan)
                                y_pred = mlp_norm_gan.predict(X_test1)
                                fpr,tpr,threshold = roc_curve(np.array([get_binary_class(yi, threshold=6.3) for yi in y_test]), y_pred)
                                roc_auc1 = auc(fpr,tpr)
                                corr1=stats.pearsonr(y_test,y_pred)
                                r2_score1 = metrics.r2_score(y_test, y_pred)
                                mse1 = metrics.mean_squared_error(y_test, y_pred)
                                y_actual_500nM = [get_binary_class(yi, threshold=6.3) for yi in y_test]
                                y_pred_500nM =[get_binary_class(yi, threshold=6.3) for yi in y_pred]
                                sen1, spec1 = perf_measure(y_actual_500nM, y_pred_500nM)
                                ap1 = average_precision_score([get_binary_class(ii) for ii in y_test], y_pred)                                                                                    
                                print({ckpt_dir1:[roc_auc1,ap1,corr1[0],r2_score1,mse1,sen1,spec1]})

                                if sen1>=0.3: 
                                    print("Save file"+ckpt_dir1) 
                                    plot_image(samples,step)
                                    os.makedirs(ckpt_dir1)
                                    saver.save(sess, os.path.join(ckpt_dir1, "model.ckpt"), global_step=step)                                    
                                    checkpoint=checkpoint+1
                        if checkpoint==5:
                            break

        plt.subplots(figsize=(20,7))
        losses=np.array(losses)
        plt.plot(losses.T[0],label="Discriminator_pos Loss")
        plt.plot(losses.T[1],label="Discriminator_neg Loss")
        plt.plot(losses.T[3],label="Generator Loss")
        plt.title("Training losses")
        plt.xlabel('Iteraions')
        plt.ylabel('Loss')
        plt.legend()
           
            
