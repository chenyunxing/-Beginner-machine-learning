from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open(r'D:\python\test.csv','r')
reader = csv.reader(allElectronicsData)
headers = next(reader)

# print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append( row[len(row) - 1])
    rowDict={}
    for i in range(1 , len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

# print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# print(str(dummyX))
# print(vec.get_feature_names())

# print(str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print(str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
# print(str(clf))

with open("a.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file = f)

# 转化dot文件为pdf文件为命令行运行
# dot -Tpdf ins.dot -o output.pdf

oneRowX = dummyX[0 , : ]
print(str(oneRowX))

newRowX = [oneRowX,]
newRowX[0][0] = 1
newRowX[0][2] = 0
print(newRowX)

predictedY = clf.predict(newRowX)
print(str(predictedY))

