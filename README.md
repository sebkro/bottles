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

--> Mit diesem Ansatz haben wir eine Accuracy von 94% erreicht - das war leider zu schön um wahr zu sein. Im Vergleich 
mit einigen neu erzeugten Testdatzen sind wir nämlich nur auf 64% gekommen. 
Die Vermutung: Das Netz hat vielleicht auf falsche Merkmale trainiert, da positive und negative Daten nacheinander erstellt wurden.

Vorgehen 2:
Das nächste Trainings-/Testset wurde erstellt, indem immer zwei Bilder mit identischer Kastenkonstellation mit 
zusätzlichen falsch plazierten Flaschen für das 'bad'-Set erstellt wurden.
 





