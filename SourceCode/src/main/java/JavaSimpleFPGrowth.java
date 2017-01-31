package my.spark;

import java.util.Arrays;
import java.util.List;
import java.io.*;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;

import org.apache.spark.SparkConf;

public class JavaSimpleFPGrowth {

  public static void main(String[] args)throws IOException {

    File file =new File("out.txt");
    file.createNewFile();
    long starttime = System.nanoTime(); 
    FileWriter writer = new FileWriter(file);
    String logFile = "/home/vpinnaka/Spark-Example/FP-Growth/sample-fpgrowth.txt";
    JavaSparkContext sc = new JavaSparkContext("local", "Simple FPGrowth",
      "/home/vpinnaka/Spark-Example/FP-Growth",
      new String[]{"target/first-example-1.0-SNAPSHOT.jar"});
    JavaRDD<String> data = sc.textFile(logFile).cache();

    JavaRDD<List<String>> transactions = data.map(
      new Function<String, List<String>>() {
        public List<String> call(String line) {
          String[] parts = line.split(",");
          return Arrays.asList(parts);
        }
      }
    );

    FPGrowth fpg = new FPGrowth()
      .setMinSupport(0.2)
      .setNumPartitions(10);
    FPGrowthModel<String> model = fpg.run(transactions);

    for (FPGrowth.FreqItemset<String> itemset: model.freqItemsets().toJavaRDD().collect()) {
      writer.write("[" + itemset.javaItems() + "], " + itemset.freq()+"\n");
    }
     writer.write("\n\n");
    double minConfidence = 0.8;
    for (AssociationRules.Rule<String> rule
      : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
      writer.write(
        rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence()+"\n");
    }
    long endtime = System.nanoTime();

    long duration = (endtime-starttime)/1000000;
    writer.write("\nExecution time :"+duration+" ms");
    writer.flush();
    writer.close();
    sc.stop();
  }
}

