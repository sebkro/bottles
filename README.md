__Flaschenerkennung__

Summary:
In diesem Projekt wird ein Neurales Netz trainiert um den Zustand sortiert/unsortiert unseres Flaschenlagers zu erkennen.

Vorgehen:
- Damit der Rechenaufwand des Trainings möglichst gering gehalten werden kann, setzen wir auf ein vorgelertes VGG16 Netzwerk auf.
- Die 100 Trainingsbilder der Klassen (gut/bad) werden verwendet um die Aktivierungen des fc2 Layers zu ermitteln und 
    abzuspeichern (FreaturizedPreSafe)
- Die letzte SoftMax Schicht wird anschließend gegen die Aktivierungen von fc2 gelernt. (FitFromFeaturized) und das 
gesamte Model gespeichert.
- Im letzten Schritt wird das Modell geladen (LoadBottleComputationGraph) und die Bilder des bisher zurückgehaltenen 
Test-Sets werden geprüft.






