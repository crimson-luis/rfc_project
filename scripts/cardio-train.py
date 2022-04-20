import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

cardio_df = pd.read_csv(
    "C:/Users/luisg/OneDrive/Documentos/Python_Scripts/rfc_project/data/raw/ENTRY1T.csv"
)
print(cardio_df.shape)
print(cardio_df.columns)
i = 1
for column in cardio_df.columns:
    print(f"# {i}. {column} - {pd.unique(cardio_df[column])}")
    i += 1

# The columns are all categorical data:
# 1. BasicGroup - ['B' 'C' 'A']
# 2. GroupCode - ['RGC' 'RGI' 'PG' 'NSG' 'NG']
# 3. AgeRange - ['45-49' '35-39' '50-54' '40-44']
# 4. MaritalStatus - ['married' 'single' 'divorced' 'widowed']
# 5. Education - ['secondary school' 'university' 'apprentice school' 'basic school' 'NS']
# 6. JobResponsibility - ['managerial worker' 'partly independent worker' 'others' 'NS'
# 'pensioner- other reasons' 'pensioner because of ICHS']
# 7. PhysActInJob - ['mainly sits' 'mainly walks' 'NS' 'carries heavy loads'
# 'mainly stands']
# 8. PhysActAfterJob - ['moderate activity' 'mainly sits' 'NS' 'great activity']
# 9. Transport - ['public' 'on foot' 'NS' 'car' 'bike']
# 10. JobTranspDuration - ['around 1 hour' 'around 1/2 hour' 'NS' 'around 2 hours'
# 'more than 2 hours']
# 11. Smoking - ['15-20 cig/day' 'non-smoker' 'cigars/pipes' '21-more cig/day'
# '5-14 cig/day' '1-4 cig/day' 'NS']
# 12. SmokingDuration - ['21 and more years' 'non-smoker' 'up to 5 years' '11-20 years'
# '6-10 years']
# 13. ExSmoker - ['keeps smoking' 'never smoked' 'yes, more than 1 year'
# 'yes, less than 1 year']
# 14. Alcohol - ['occasionally' 'regulary' 'NS' 'no']
# 15. Beer7 - ['no' 'yes']
# 16. Beer10 - ['no' 'yes']
# 17. Beer12 - ['no' 'yes']
# 18. Beer - ['no' 'yes']
# 19. Wine - ['no' 'yes']
# 20. Liquor - ['yes' 'no']
# 21. DailyBeerCons - ['does not drink beer' 'up to 1 litre' 'does not drink alcohol'
# 'more than 1 litre' 'NS']
# 22. DailyWineCons - ['does not drink wine' 'up to half a litre' 'does not drink alcohol'
# 'more than half a litre' 'NS']
# 23. DailyLiquorCons - ['more than 100cc' 'does not drink liquor' 'up to 100cc'
# 'does not drink alcohol' 'NS']
# 24. HeightRange - ['1.60 - 1.69' '1.70 - 1.79' '> 1.79' 'NS' '< 1.60']
# 25. WeightRange - ['70 - 79' '80 - 89' '90 - 99' '60 - 69' '100 - 109' '110 - 119'
# '> 119'
# '< 60' 'NS']
# 26. BMIRange - ['normal' 'overweight' 'obese' 'NS' 'underweight' 'morbidly obese']
# 27. BloodPressure - ['High' 'Normal' 'Normal/High']
# 28. SkinfoldTric - ['4' '10' '12' '18' '15' '13' '8' 'NS' '17' '5' '9' '14' '7' '6' '3'
# '16' '11' '28' '19' '21' '32' '26' '20' '22' '1' '2' '24' '23' '25' '35' '29']
# 29. SkinfoldSubsc - ['12' '22' '15' '20' '26' '35' '39' '18' '14' '32' 'NS' '13' '23'
# '16' '8' '17' '11' '24' '19' '45' '30' '25' '31' '46' '44' '37' '42' '21' '33'
# '10' '50' '34' '36' '41' '6' '29' '9' '28' '7' '38' '27' '4' '40' '5'
# '49' '47' '48' '70']
# 30. Cholesterol - ['bordering' 'desirable' 'high' 'NS']
# 31. Triglycerides - ['desirable' 'NS' 'bordering' 'high' 'very high']
# 32. Urine - ['normal' 'albumen positive' 'NS' 'sugar positive']

print(cardio_df.describe())

cardio_df.dropna(inplace=True)

sns.set_style("darkgrid")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (9, 5)
matplotlib.rcParams["figure.facecolor"] = "#00000000"


# The graph below presents the distribution of age in the dataset. There is no patient
# younger than 30 years old and most of them are between 50 and 60 years old.


plt.figure(figsize=(12, 6))
plt.title("Distribution of age")
plt.xlabel("Age")
plt.ylabel("Number of patients")

plt.hist(x="AgeRange", color="purple", data=cardio_df)

# The distribution of gender in this dataset is not expected, the majority are women,
# which is not common, since they were supposed to be randomly selected.



plt.figure(figsize=(12, 6))
plt.title("Distribution of BasicGroup")
plt.pie(
    cardio_df.BasicGroup.value_counts(),
    labels=("A", "B", "C"),
    autopct="%1.1f%%",
    startangle=180,
)

# As expected, there is a positive linear relation between height and weight.
# On the other hand, this relation could be stronger if the dataset had more reliable
# information.



plt.figure(figsize=(12, 6))
plt.title("Distribution of age")
plt.xlabel("Age")
plt.ylabel("Number of patients")

plt.hist(x="HeightRange", color="purple", data=cardio_df)

plt.figure(figsize=(12, 6))
plt.title("Distribution of age")
plt.xlabel("Age")
plt.ylabel("Number of patients")

plt.hist(x="WeightRange", color="purple", data=cardio_df)

# Plotting the cholesterol level, it is evident that most patients have a normal level
# (level 1).


cardio_df.cholesterol.value_counts()

sns.countplot(y=cardio_df.cholesterol)
plt.xticks(rotation=75)
plt.title("Cholesterol level of patients")
plt.ylabel(None)

# As shown in the graph below, normal glucose level is most common between patients.


cardio_df.gluc.value_counts()

sns.countplot(y=cardio_df.gluc)
plt.xticks(rotation=75)
plt.title("Glucose level of patients")
plt.ylabel(None)

# ## Asking and Answering Questions
#
# After the insights about the patients by exploring individual columns of the dataset.
# Now, the objective is to ask some
# specific questions and try to answer them using data frame operations and
# visualizations.

# #### Q1: Among patients who had cardiovascular disease, how many smoked?

# To answer this question, a value_counts and a dataframe filter will be used.


x1 = cardio_df[cardio_df.cardio == 1].smoke
plt.figure(figsize=(12, 6))
plt.title("Patients that have cardio disease")
plt.pie(
    x1.value_counts(),
    labels=("Do not smoke", "Smoke"),
    autopct="%1.1f%%",
    startangle=180,
)

x2 = cardio_df[cardio_df.smoke == 1].cardio
plt.figure(figsize=(12, 6))
plt.title("Patients that smokes")
plt.pie(
    x2.value_counts(),
    labels=("Do not have", "Have cardio disease"),
    autopct="%1.1f%%",
    startangle=180,
)

# As shown above, only eight percent of patients smoked. On the other hand, 47% of
# patients that smoked had a cardiovascular disease. So, it is common to have cardio
# disease if you smoke, but this indicator is not a very representative cause.

# #### Q2: What is the most common age range among patients with heart disease?

# To see this information properly, it is used a groupby at the age column counting the
# binary cardio column.


data = cardio_df.groupby(["years_old"]).count().cardio

plt.figure(figsize=(12, 12))
ax = sns.barplot(x=data.index.astype(int), y=data)
plt.title("Patients with cardio disease frequency by age")
plt.xlabel("Age")
plt.ylabel("Frequency")

# It is highlighted in the graph above that there is a concentration of people with
# disease between 50 and 60 years old.

# #### Q3: How can we assure that is a relation between height and weight in this dataset?

# This can be done using a regression model to draw a trend line. In this case, both the
# numpy.polyfit or the sns.lmplot packages can be used. For a linear model the order
# parameter is set to one and hue parameter can be used to create a dummy variable of a
# certain column.


sns.lmplot(x="height", y="weight", data=cardio_df, order=1, hue="gender")

# Fitted values by the function shows that is reasonable to believe that there is indeed
# a expected relation between these two variables in this dataset.

# #### Q4: Is there a difference in blood pressure variables between patients with or
# without heart disease?

# To answer this question we can get a sample from each condition and then test their
# mean using a statistical method.

ap_lo_c = cardio_df[cardio_df.cardio == 1].sample(1000).ap_lo
ap_lo_n = cardio_df[cardio_df.cardio == 0].sample(1000).ap_lo
print(
    "Mean of systolic pressure of patients with disease",
    ap_lo_c.mean(),
    "\nMean of systolic pressure of patients without disease",
    ap_lo_n.mean(),
)
stat, p = mannwhitneyu(ap_lo_c, ap_lo_n)
print("\nstat=%.3f, p=%.3f" % (stat, p))
if p > 0.05:
    print("\nProbably the same distribution")
else:
    print("\nProbably different distributions")

ap_hi_c = cardio_df[cardio_df.cardio == 1].sample(1000).ap_hi
ap_hi_n = cardio_df[cardio_df.cardio == 0].sample(1000).ap_hi
print(
    "Mean of diastolic pressure of patients with disease",
    ap_hi_c.mean(),
    "\nMean of diastolic pressure of patients without disease",
    ap_hi_n.mean(),
)
stat, p = mannwhitneyu(ap_hi_c, ap_hi_n)
print("\nstat=%.3f, p=%.3f" % (stat, p))
if p > 0.05:
    print("\nProbably the same distribution")
else:
    print("\nProbably different distributions")

# The results show that it is reasonable to believe that the distribution of blood
# pressure variables are different between patients with and without heart disease, with
# higher values in patients with the disease.

# #### Q1: Among patients who had cardiovascular disease, how many drink?


x1 = cardio_df[cardio_df.cardio == 1].alco
plt.figure(figsize=(12, 6))
plt.title("Patients that have cardio disease")
plt.pie(
    x1.value_counts(),
    labels=("Do not drink", "Drink"),
    autopct="%1.1f%%",
    startangle=180,
)

x2 = cardio_df[cardio_df.smoke == 1].alco
plt.figure(figsize=(12, 6))
plt.title("Patients that drink")
plt.pie(
    x2.value_counts(),
    labels=("Do not have", "Have cardio disease"),
    autopct="%1.1f%%",
    startangle=180,
)
