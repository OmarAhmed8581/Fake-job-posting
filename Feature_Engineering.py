import pandas as pd

def Feature_Engineering(dataset_path):
    fake_job_posting = pd.read_csv(dataset_path)

    # # ** ======== Cleaning Data ================== **
    print("\n*==== Applying Featuring Engineering ====*\n")
    # print(fake_job_posting.isnull().sum())
    # # Drop data
    # drop a Fireplace table because more than 50% data are null.
    new_1_fake_job_posting = fake_job_posting.drop("salary_range", axis=1)
    new_2_fake_job_posting = new_1_fake_job_posting.drop("department", axis=1)
    new_3_fake_job_posting = new_2_fake_job_posting.drop("required_education", axis=1)

    # Now drop unnecessary column  .

    new_4_fake_job_posting = new_3_fake_job_posting.drop("title", axis=1)
    new_5_fake_job_posting = new_4_fake_job_posting.drop("benefits", axis=1)
    new_6_fake_job_posting = new_5_fake_job_posting.drop("location", axis=1)
    new_7_fake_job_posting = new_6_fake_job_posting.drop("requirements", axis=1)
    new_8_fake_job_posting = new_7_fake_job_posting.drop("company_profile", axis=1)
    new_9_fake_job_posting = new_8_fake_job_posting.drop("description", axis=1)
    new_10_fake_job_posting = new_9_fake_job_posting.drop("industry", axis=1)



    # now fill null values
    new_11_fake_job_posting = new_10_fake_job_posting.fillna(method='bfill')


    # print(new_5_fake_job_posting.info())

    # # Dummies is used to convert a string into binaray beacuse we want to identify 1 0 that this data is exist ot not while
    # #we used dummies method
    new_12_fake_job_posting = pd.get_dummies(new_11_fake_job_posting, columns=['employment_type'], drop_first=True,
                                             prefix='employment')
    new_13_fake_job_posting = pd.get_dummies(new_12_fake_job_posting, columns=['required_experience'], drop_first=True,
                                             prefix='required_experience')
    new_14_fake_job_posting = pd.get_dummies(new_13_fake_job_posting, columns=['function'], drop_first=True,
                                             prefix='function')

    print(new_14_fake_job_posting.info)




    return  new_14_fake_job_posting





