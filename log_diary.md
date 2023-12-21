|Daily unsolved|[Dec 11th]|
| :- | -: |
|||

**Issue1:** Unable to group the rows in weeks

**Solution1:** Cluster the rows
    <df.groupby(["endtime"])["behavior_scale"].median()>

**Issue2:** The box plot reproduced from the paper does not make sense

**Solution2:** The x axis is not in time order

**Issue3:** Failed to sort the index of the groupby in time order

**Solution3:** Convet it to the date type <df["endtime"] = pd.to_datetime(df["endtime"])>
Then sort the dataframe before groupby <df = df.sort_values(by=["endtime"])>

**Issue4:** Not able to drop the columns with NA
**Solution4:** <df.dropna(inplace = True)>

**To do tmr:** Plot scatter or box plot for grouped week for frequency. For example each week has an average of the frequency or box.