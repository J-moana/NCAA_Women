
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from subprocess import check_output


print(check_output(["ls", "WDataFiles"]).decode("utf8"))


data_dir = 'WDataFiles/'
# historical seeding talbe
df_seeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
# result of tour
df_tour = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')

# print(df_seeds.head())
# print(df_tour.head())

# take seed # only, remove the area code.
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
# print("seed len", len(df_seeds))
# print(df_seeds.head(100))

df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
# print(df_tour.head())

df_winseeds = df_seeds.rename(columns = {'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns = {'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left = df_tour,right= df_winseeds, on=['Season', 'WTeamID'])
# print(df_dummy.head())

# make a table w/ Wteam - Lteam - and their seed numbers
df_concat = pd.merge(left = df_dummy,right= df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
# print(df_concat.head())

df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins,df_losses))
print(df_predictions.head())

X_train = df_predictions.SeedDiff.values.reshape(-1,1)

y_train = df_predictions.Result.values
X_train,y_train = shuffle(X_train,y_train)

## training the model

logreg = LogisticRegression()
params = {'C':np.logspace(start=-5,stop=3,num=9)}
clf = GridSearchCV(logreg,params,scoring='neg_log_loss',refit=True)
clf.fit(X_train,y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

X = np.arange(-10,10).reshape(-1,1)
preds = clf.predict_proba(X)[:,1]


plt.plot(X,preds)
plt.xlabel('Team1 seed -Team2 seed')
plt.ylabel('P(Team1 will win)')
plt.show()

df_sample_sub = pd.read_csv('WSampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games,1))

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii,0] = diff_seed


preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
print(df_sample_sub.head(30))


df_sample_sub.to_csv('logreg_seed_starter.csv', index=False)
