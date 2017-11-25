package org.hackathon.bottles;

public class Configuration {

    public static void main(String[] args){
        System.out.println(Configuration.baseFolder());
    }
    public static String baseFolder(){
        return System.getenv("BASE_FOLDER");
    }
}
