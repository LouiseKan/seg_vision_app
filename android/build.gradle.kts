allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    val projectPath = project.path
    if (projectPath.contains(":app")) {
        evaluationDependsOn(":app")
    }
    // 剛才加的那段 afterEvaluate 全部刪掉，我們已經手動修好套件了
}
tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

