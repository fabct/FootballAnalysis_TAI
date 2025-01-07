
import { HttpClientModule } from '@angular/common/http'
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { FormsModule } from '@angular/forms';

import { HeaderComponent } from './header/header.component';
import { AdminComponent } from './admin/admin.component';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatSortModule } from '@angular/material/sort';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { AppRoutingModule } from './app-routing.module';
import { VideouploadComponent } from './videoupload/videoupload.component';
import { VideodownloadComponent } from './videodownload/videodownload.component';

@NgModule({
  imports: [
    BrowserModule,
    FormsModule,
    HttpClientModule,
    MatSortModule,
    MatTableModule,
    MatPaginatorModule,
    BrowserAnimationsModule,
    AppRoutingModule

  ],
  declarations: [
    AppComponent,
    LoginComponent,
    RegisterComponent,

    HeaderComponent,
    AdminComponent,
    VideouploadComponent,
    VideodownloadComponent,
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }