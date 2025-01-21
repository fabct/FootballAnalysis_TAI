import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { VideodownloadComponent } from './videodownload/videodownload.component'; // Importez votre composant ici
import { VideouploadComponent } from './videoupload/videoupload.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from './auth.guard';

import { AppComponent } from './app.component';
const routes: Routes = [
  { path: 'videodownload', component: VideodownloadComponent},
  { path: '', component: VideouploadComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
