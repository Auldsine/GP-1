package com.example.allsafe;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.Manifest;

public class MainActivity extends AppCompatActivity {

    private TextView myTextView;
    private Button nextButton, prevButton;
    private Handler handler = new Handler();
    private Cursor cursor;
    private int messageIndex;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        myTextView = findViewById(R.id.textView);
        nextButton = findViewById(R.id.nextButton);
        prevButton = findViewById(R.id.prevButton);


        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_SMS}, PackageManager.PERMISSION_GRANTED);

        READ_SMS();
    }

    public void READ_SMS(){

        cursor = getContentResolver().query(Uri.parse("content://sms"), null, null, null, "date DESC");

        if (cursor != null && cursor.moveToFirst()) {
            messageIndex = 0;
            displayMessage();
        }

    }

    public void nextMessage(View view) {
        if (cursor != null && cursor.moveToNext() && messageIndex < cursor.getCount() - 1) {

            messageIndex++;
            displayMessage();
        }
    }

    public void prevMessage(View view) {
        if (cursor != null && cursor.moveToPrevious() && messageIndex > 0) {
            messageIndex--;
            displayMessage();
        }
    }

    private void displayMessage() {
        if (cursor != null){
            String address = cursor.getString(cursor.getColumnIndexOrThrow("address"));
            String message = cursor.getString(cursor.getColumnIndexOrThrow("body"));

            StringBuilder smsBuilder = new StringBuilder();

            smsBuilder.append("From: ");
            smsBuilder.append(address);
            smsBuilder.append("\n");
            smsBuilder.append("Message:\n");
            smsBuilder.append(message);
            smsBuilder.append("\n");
            smsBuilder.append("______________________________________");
            smsBuilder.append("\n \n");

            myTextView.setText(smsBuilder.toString());

            if (messageIndex == cursor.getCount() - 1) {
                nextButton.setEnabled(false);
            } else {
                nextButton.setEnabled(true);
            }

            if (messageIndex == 0) {
                prevButton.setEnabled(false);
            } else {
                prevButton.setEnabled(true);
            }
        }
    }
}
